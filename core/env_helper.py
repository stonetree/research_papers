# -*- coding: utf-8 -*-
import os
import platform

def get_env_var(name: str, default: str = None) -> str:
    """
    获取环境变量，具备 Windows 注册表实时打捞与本地 .env 文件手动解析的顽强韧性。
    优先读取 os.environ，在 Windows 系统下如果读取不到，则尝试从注册表实时获取。
    同时，如果项目根目录下存在 .env 文件，也支持手动读取解析。
    """
    # 1. 优先从当前进程的环境变量读取
    val = os.environ.get(name)
    if val:
        return val.strip()

    # 2. 如果是 Windows 系统，尝试从注册表读取（解决因环境变量更新而未重启进程导致的读取失败）
    if platform.system() == "Windows":
        import winreg
        # 尝试 HKCU
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                val, _ = winreg.QueryValueEx(key, name)
                if val:
                    return val.strip()
        except Exception:
            pass
        # 尝试 HKLM
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment") as key:
                val, _ = winreg.QueryValueEx(key, name)
                if val:
                    return val.strip()
        except Exception:
            pass

    # 3. 尝试读取根目录下的 .env 文件
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == name:
                            # 移除引号
                            v_val = v.strip().strip("'").strip('"')
                            return v_val
        except Exception:
            pass

    return default
