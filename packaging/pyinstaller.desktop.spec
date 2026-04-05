# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

root = Path.cwd()

block_cipher = None

a = Analysis(
    [str(root / "app" / "main.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / "config.json"), ".")],
    hiddenimports=["app.main", "app.controller", "app.gui", "app.session", "app.capture_overlay", "app.config_store", "app.models", "app.error_handler"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BETTERME",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BETTERME",
)
