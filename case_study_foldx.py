import hashlib
import json
import os
import shutil
import subprocess
from typing import Optional, Tuple


BUILDMODEL_DDG_PARSER_VERSION = 2


class CaseStudyFoldXBuilder:
    """Case-study-only FoldX helpers kept separate from the main FoldXProcessor."""

    def __init__(self, foldx_path: str, temp_dir: str, cache_dir: str):
        self.foldx_path = os.path.abspath(foldx_path)
        self.temp_dir = os.path.abspath(temp_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.foldx_identity = (
            f"{os.path.basename(self.foldx_path)}:"
            f"{os.path.getsize(self.foldx_path) if os.path.exists(self.foldx_path) else 'missing'}"
        )
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_mutant_model(
        self,
        repaired_pdb_path: str,
        foldx_mutation_spec: str,
    ) -> str:
        mutant_path, _ = self.build_mutant_model_with_ddg(
            repaired_pdb_path=repaired_pdb_path,
            foldx_mutation_spec=foldx_mutation_spec,
        )
        return mutant_path

    def build_mutant_model_with_ddg(
        self,
        repaired_pdb_path: str,
        foldx_mutation_spec: str,
    ) -> Tuple[str, Optional[float]]:
        normalized_spec = foldx_mutation_spec.strip()
        if not normalized_spec.endswith(";"):
            normalized_spec = f"{normalized_spec};"

        cache_key = hashlib.md5(
            f"{self.foldx_identity}_build_{repaired_pdb_path}_{normalized_spec}".encode()
        ).hexdigest()
        final_mutant_path = os.path.join(self.cache_dir, f"{cache_key}_Mutant.pdb")
        metadata_path = os.path.join(self.cache_dir, f"{cache_key}_BuildModel.json")
        if os.path.exists(final_mutant_path):
            return final_mutant_path, self._read_buildmodel_ddg(metadata_path)

        sandbox_dir = os.path.join(self.temp_dir, f"case_build_{cache_key[:12]}")
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        os.makedirs(sandbox_dir, exist_ok=True)

        foldx_exe_name = os.path.basename(self.foldx_path)
        shutil.copy2(self.foldx_path, os.path.join(sandbox_dir, foldx_exe_name))

        foldx_root = os.path.dirname(self.foldx_path)
        rotabase_path = os.path.join(foldx_root, "rotabase.txt")
        if os.path.exists(rotabase_path):
            shutil.copy2(rotabase_path, os.path.join(sandbox_dir, "rotabase.txt"))

        pdb_filename = os.path.basename(repaired_pdb_path)
        shutil.copy2(repaired_pdb_path, os.path.join(sandbox_dir, pdb_filename))

        mutant_file_name = f"individual_list_{cache_key[:12]}.txt"
        with open(os.path.join(sandbox_dir, mutant_file_name), "w", encoding="utf-8") as f:
            f.write(normalized_spec + "\n")

        output_tag = f"{os.path.splitext(pdb_filename)[0]}_{cache_key[:8]}"
        cmd = [
            f"./{foldx_exe_name}",
            "--command=BuildModel",
            f"--pdb={pdb_filename}",
            f"--mutant-file={mutant_file_name}",
            "--numberOfRuns=1",
            f"--output-file={output_tag}",
            "--output-dir=.",
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=sandbox_dir,
                timeout=int(os.environ.get("FOLDX_BUILD_TIMEOUT", "900")),
            )
        except subprocess.TimeoutExpired:
            raise Exception(f"FoldX BuildModel timed out for {repaired_pdb_path}.")
        except subprocess.CalledProcessError as e:
            print(f"FoldX BuildModel failed for {repaired_pdb_path}.")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            raise

        buildmodel_ddg = self._parse_buildmodel_ddg(sandbox_dir)
        mutant_source_path = self._find_built_mutant_pdb(
            sandbox_dir=sandbox_dir,
            input_pdb_filename=pdb_filename,
        )
        if mutant_source_path is None:
            raise FileNotFoundError(
                f"Mutant PDB file not found after FoldX BuildModel in {sandbox_dir}"
            )

        shutil.copy2(mutant_source_path, final_mutant_path)
        self._write_buildmodel_metadata(
            metadata_path=metadata_path,
            repaired_pdb_path=repaired_pdb_path,
            mutation_spec=normalized_spec,
            mutant_pdb_path=final_mutant_path,
            buildmodel_ddg=buildmodel_ddg,
        )
        return final_mutant_path, buildmodel_ddg

    def _read_buildmodel_ddg(self, metadata_path: str) -> Optional[float]:
        if not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if int(payload.get("buildmodel_ddg_parser_version", 0)) < BUILDMODEL_DDG_PARSER_VERSION:
                return None
            value = payload.get("foldx_buildmodel_ddg")
            return None if value is None else float(value)
        except Exception:
            return None

    def _write_buildmodel_metadata(
        self,
        metadata_path: str,
        repaired_pdb_path: str,
        mutation_spec: str,
        mutant_pdb_path: str,
        buildmodel_ddg: Optional[float],
    ) -> None:
        payload = {
            "repaired_pdb_path": os.path.abspath(repaired_pdb_path),
            "mutant_pdb_path": os.path.abspath(mutant_pdb_path),
            "foldx_mutation_spec": mutation_spec,
            "foldx_buildmodel_ddg": buildmodel_ddg,
            "buildmodel_ddg_parser_version": BUILDMODEL_DDG_PARSER_VERSION,
            "foldx_identity": self.foldx_identity,
            "foldx_path": self.foldx_path,
        }
        tmp_path = f"{metadata_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, metadata_path)

    def _parse_buildmodel_ddg(self, sandbox_dir: str) -> Optional[float]:
        candidates = []
        for name in os.listdir(sandbox_dir):
            lower = name.lower()
            if lower.startswith("dif_") and lower.endswith(".fxout"):
                candidates.append(os.path.join(sandbox_dir, name))
        if not candidates:
            for name in os.listdir(sandbox_dir):
                lower = name.lower()
                if lower.startswith("average_") and lower.endswith(".fxout"):
                    candidates.append(os.path.join(sandbox_dir, name))

        for path in sorted(candidates):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parsed = self._parse_buildmodel_table_row(line)
                    if parsed is not None:
                        return parsed
        return None

    def _parse_buildmodel_table_row(self, line: str) -> Optional[float]:
        parts = line.split("\t") if "\t" in line else line.split()
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) < 3:
            return None

        first = parts[0].lower()
        metadata_keys = {
            "pdb",
            "ph",
            "temperature",
            "ionstrength",
            "numberofruns",
            "command",
            "time",
            "date",
            "user",
            "output-file",
        }
        if first in metadata_keys:
            return None

        numeric_values = []
        for token in parts[1:]:
            try:
                numeric_values.append(float(token))
            except ValueError:
                continue

        # Real FoldX BuildModel table rows contain multiple numeric energy terms.
        # Parameter/header lines such as "pH 5.0" must not be interpreted as ddG.
        if len(numeric_values) < 3:
            return None
        return numeric_values[0]

    def _find_built_mutant_pdb(
        self,
        sandbox_dir: str,
        input_pdb_filename: str,
    ) -> Optional[str]:
        input_pdb_path = os.path.abspath(os.path.join(sandbox_dir, input_pdb_filename))
        pdb_candidates = []

        for name in os.listdir(sandbox_dir):
            if not name.lower().endswith(".pdb"):
                continue
            full_path = os.path.abspath(os.path.join(sandbox_dir, name))
            if full_path == input_pdb_path:
                continue
            pdb_candidates.append(full_path)

        if not pdb_candidates:
            return None

        preferred = [
            path for path in pdb_candidates
            if not os.path.basename(path).startswith("WT_")
        ]
        if preferred:
            pdb_candidates = preferred

        pdb_candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        return pdb_candidates[0]
