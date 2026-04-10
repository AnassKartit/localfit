"""py2app setup for LocalFit.app — macOS menu bar application."""
from setuptools import setup

APP = ["src/localfit/menubar.py"]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "assets/localfit.icns",
    "plist": {
        "CFBundleIdentifier": "com.localfit.app",
        "CFBundleName": "LocalFit",
        "CFBundleDisplayName": "LocalFit",
        "CFBundleShortVersionString": "0.4.0",
        "CFBundleVersion": "0.4.0",
        "LSUIElement": True,  # Menu bar only — no dock icon
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    },
    "packages": ["localfit", "rich", "prompt_toolkit"],
    "includes": ["rumps"],
    "excludes": ["textual", "huggingface_hub"],  # Keep .app size down
}

setup(
    name="LocalFit",
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
