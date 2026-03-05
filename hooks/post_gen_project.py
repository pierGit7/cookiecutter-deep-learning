import os
import shutil
from pathlib import Path

def main():
    # Remove LICENSE if "No license file"
    if "{{ cookiecutter.open_source_license }}" == "No license file":
        if Path("LICENSE").exists():
            Path("LICENSE").unlink()

    # Make single quotes prettier
    # Jinja tojson escapes single-quotes with \u0027 since it's meant for HTML/JS
    if Path("pyproject.toml").exists():
        pyproject_text = Path("pyproject.toml").read_text()
        Path("pyproject.toml").write_text(pyproject_text.replace(r"\u0027", "'"))

if __name__ == "__main__":
    main()
