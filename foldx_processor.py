import os
import subprocess
import re
import hashlib
import pickle
from typing import List, Optional
import threading
import time



class FoldXProcessor:
    def __init__(
        self, 
        foldx_path: str = "./foldx", 
        temp_dir: str = "./foldx_temp",
        cache_dir: str = "./foldx_cache",
        cache_size: int = 5000  # 缓存最大条目数
    ):
        self.cache_cleaner_thread = threading.Thread(target=self._background_cache_cleaner, daemon=True)
        self.cache_cleaner_thread.start()
        self.foldx_path = os.path.abspath(foldx_path)
        self.temp_dir = os.path.abspath(temp_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.foldx_identity = (
            f"{os.path.basename(self.foldx_path)}:"
            f"{os.path.getsize(self.foldx_path) if os.path.exists(self.foldx_path) else 'missing'}"
        )
        self.repair_timeout = int(os.environ.get("FOLDX_REPAIR_TIMEOUT", "300"))
        self.analyse_timeout = int(os.environ.get("FOLDX_ANALYSE_TIMEOUT", "300"))
        self.cache_size = cache_size  # 缓存大小限制
        
        # 创建必要目录
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化缓存
        self._cache = {}
        self._cache_file = os.path.join(self.cache_dir, "foldx_cache.pkl")
        self._load_cache()

    def _background_cache_cleaner(self):
        """后台线程定期清理过期缓存（每小时执行一次）"""
        while True:
            time.sleep(3600)  # 1小时
            self._trim_cache()
            self._save_cache()

    def _get_cache_key(self, input_str: str) -> str:
        """生成缓存键（使用MD5哈希）"""
        return hashlib.md5(input_str.encode()).hexdigest()

    def _load_cache(self) -> None:
        """从磁盘加载缓存"""
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                # 确保缓存不超过设定大小
                if len(self._cache) > self.cache_size:
                    self._trim_cache()
            except (pickle.UnpicklingError, EOFError, IOError) as e:
                print(f"Warning: Failed to load cache - {e}. Starting with empty cache.")
                self._cache = {}

    def _save_cache(self) -> None:
        """将缓存保存到磁盘"""
        try:
            # 保存前先修剪缓存
            self._trim_cache()
            with open(self._cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except IOError as e:
            print(f"Warning: Failed to save cache - {e}")

    def _trim_cache(self) -> None:
        """如果缓存超过最大大小，删除最早的条目"""
        if len(self._cache) > self.cache_size:
            # 按插入顺序保留最新的条目
            keys_to_keep = list(self._cache.keys())[-self.cache_size:]
            self._cache = {k: self._cache[k] for k in keys_to_keep}

    def _parse_interaction_energy_from_stdout(
        self, stdout: str, partner1_chains: List[str], partner2_chains: List[str]
    ) -> float:
        total_interaction_energy = 0.0
        chain_pairs = [(c1, c2) for c1 in partner1_chains for c2 in partner2_chains]

        for c1, c2 in chain_pairs:
            pattern = re.compile(
                rf"interaction between {c1} and {c2}|interaction between {c2} and {c1}"
            )
            match = pattern.search(stdout)

            if match:
                block = stdout[match.end() :]
                total_match = re.search(r"Total\s+=\s+([\-0-9.]+)", block)
                if total_match:
                    total_interaction_energy += float(total_match.group(1))

        if not chain_pairs:
            print("Warning: No chain pairs provided to parse interaction energy.")
            return 0.0

        return total_interaction_energy

    def preprocess_pdb(self, pdb_path: str) -> str:
        """修复PDB文件并使用缓存避免重复处理"""
        # 生成缓存键（基于原始PDB路径）
        cache_key = self._get_cache_key(f"{self.foldx_identity}_preprocess_{pdb_path}")
        
        # 检查缓存
        if cache_key in self._cache:
            repaired_path = self._cache[cache_key]
            if os.path.exists(repaired_path):
                return repaired_path

        # 缓存未命中，执行修复
        import shutil
        
        # 准备沙盒目录
        sandbox_dir = self.temp_dir # 假设 temp_dir 已经是唯一的沙盒目录
        
        # 复制 FoldX 可执行文件
        foldx_exe_name = os.path.basename(self.foldx_path)
        sandbox_foldx_path = os.path.join(sandbox_dir, foldx_exe_name)
        if not os.path.exists(sandbox_foldx_path):
            shutil.copy2(self.foldx_path, sandbox_foldx_path)
        os.chmod(sandbox_foldx_path, 0o755)
            
        # 复制 rotabase.txt (如果在 foldx 同级目录)
        foldx_root = os.path.dirname(self.foldx_path)
        rotabase_path = os.path.join(foldx_root, "rotabase.txt")
        if os.path.exists(rotabase_path):
            shutil.copy2(rotabase_path, os.path.join(sandbox_dir, "rotabase.txt"))
            
        # 复制目标 PDB 文件
        pdb_filename = os.path.basename(pdb_path)
        sandbox_pdb_path = os.path.join(sandbox_dir, pdb_filename)
        shutil.copy2(pdb_path, sandbox_pdb_path)

        cmd = [
            f"./{foldx_exe_name}",
            "--command=RepairPDB",
            f"--pdb={pdb_filename}",
            "--output-dir=.", # 输出到当前目录
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=sandbox_dir,
                timeout=self.repair_timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"FoldX RepairPDB timed out for {pdb_path}.")
            # 尝试杀死子进程（虽然 timeout 应该会自动处理，但为了安全）
            raise Exception(f"FoldX RepairPDB timed out for {pdb_path}")
        except subprocess.CalledProcessError as e:
            print(f"FoldX RepairPDB failed for {pdb_path}.")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            raise e
            
        repaired_pdb_filename = f"{os.path.splitext(pdb_filename)[0]}_Repair.pdb"
        sandbox_repaired_path = os.path.join(sandbox_dir, repaired_pdb_filename)

        if not os.path.exists(sandbox_repaired_path):
            raise FileNotFoundError(
                f"Repaired PDB file not found after running FoldX. Expected at: {sandbox_repaired_path}"
            )
        
        # 将修复后的 PDB 移动到统一的缓存目录，以免被清理
        # 注意：这里我们可能需要一个持久的 cache_dir 来存放修复后的 PDB
        # 或者直接返回 sandbox 中的路径（但 sandbox 可能会被清理）
        # 这里为了简单，我们把修复后的 PDB 复制回 self.cache_dir
        
        # 使用哈希命名以避免冲突
        final_repaired_name = f"{cache_key}_Repair.pdb"
        final_repaired_path = os.path.join(self.cache_dir, final_repaired_name)
        shutil.copy2(sandbox_repaired_path, final_repaired_path)
        
        # 更新缓存
        self._cache[cache_key] = final_repaired_path
        self._save_cache()
        
        return final_repaired_path

    def extract_features(
        self,
        repaired_pdb_path: str,
        partner1_chains: List[str],
        partner2_chains: List[str],
    ) -> float:
        """提取相互作用能并使用缓存避免重复计算"""
        # 生成缓存键（基于修复后的PDB路径和链信息）
        chains_key = "_".join(sorted(partner1_chains + partner2_chains))
        cache_key = self._get_cache_key(f"{self.foldx_identity}_extract_{repaired_pdb_path}_{chains_key}")
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 缓存未命中，执行计算
        import shutil
        
        # 准备沙盒目录
        sandbox_dir = self.temp_dir
        
        # 复制 FoldX 可执行文件
        foldx_exe_name = os.path.basename(self.foldx_path)
        sandbox_foldx_path = os.path.join(sandbox_dir, foldx_exe_name)
        if not os.path.exists(sandbox_foldx_path):
            shutil.copy2(self.foldx_path, sandbox_foldx_path)
        os.chmod(sandbox_foldx_path, 0o755)
            
        # 复制 rotabase.txt
        foldx_root = os.path.dirname(self.foldx_path)
        rotabase_path = os.path.join(foldx_root, "rotabase.txt")
        if os.path.exists(rotabase_path):
            shutil.copy2(rotabase_path, os.path.join(sandbox_dir, "rotabase.txt"))
            
        # 复制 PDB 文件
        pdb_filename = os.path.basename(repaired_pdb_path)
        sandbox_pdb_path = os.path.join(sandbox_dir, pdb_filename)
        shutil.copy2(repaired_pdb_path, sandbox_pdb_path)

        # 构建 chains 参数 (e.g., "A,B")
        chains_arg = f"{''.join(partner1_chains)},{''.join(partner2_chains)}"
        
        energy_cmd = [
            f"./{foldx_exe_name}",
            "--command=AnalyseComplex",
            f"--pdb={pdb_filename}",
            f"--analyseComplexChains={chains_arg}"
        ]

        try:
            process_result = subprocess.run(
                energy_cmd,
                capture_output=True,
                text=True,
                cwd=sandbox_dir,
                timeout=self.analyse_timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"FoldX AnalyseComplex timed out for {repaired_pdb_path}")
            return 0.0

        if process_result.returncode != 0:
            print(f"FoldX AnalyseComplex failed for {repaired_pdb_path}.")
            print(f"Stderr: {process_result.stderr}")
            print(f"Stdout: {process_result.stdout}")
            # 不抛出异常，而是返回 0.0，避免中断整个流程
            return 0.0

        # 解析输出文件
        pdb_basename_no_ext = os.path.splitext(pdb_filename)[0]
        summary_file = os.path.join(sandbox_dir, f"Summary_{pdb_basename_no_ext}_AC.fxout")
        
        interaction_energy = 0.0
        if os.path.exists(summary_file):
            interaction_energy = self._parse_interaction_energy_from_file(summary_file)
        else:
            # 尝试从 stdout 解析 (fallback)
            interaction_energy = self._parse_interaction_energy_from_stdout(
                process_result.stdout, partner1_chains, partner2_chains
            )
        
        # 如果解析失败（为0.0），记录警告
        if interaction_energy == 0.0:
            print(f"Warning: Interaction energy is 0.0 for {repaired_pdb_path}")

        # 更新缓存
        self._cache[cache_key] = interaction_energy
        self._save_cache()
        
        return interaction_energy

    def _parse_interaction_energy_from_file(self, file_path: str) -> float:
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            # 查找数据行
            # Pdb Group1 Group2 IntraclashesGroup1 IntraclashesGroup2 Interaction Energy ...
            header_index = -1
            for i, line in enumerate(lines):
                if "Interaction Energy" in line:
                    header_index = i
                    break
            
            if header_index != -1 and header_index + 1 < len(lines):
                data_line = lines[header_index + 1]
                parts = data_line.split()
                # Interaction Energy 通常是第6列 (index 5)
                # ./pdb E I 28.3 6.7 -12.6 ...
                if len(parts) >= 6:
                    return float(parts[5])
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        return 0.0
