import os
import sys
from pathlib import Path
from typing import Optional

class CloudStorage:
    """
    Handles seamless path resolution between Local (VS Code) and Google Colab environments.
    """
    
    def __init__(self, project_name: str = "surgical_pruning"):
        self.project_name = project_name
        self.is_colab = 'google.colab' in sys.modules
        self.drive_mounted = False

    def mount_drive(self, mount_point: str = "/content/drive"):
        """Mounts Google Drive if running in Colab."""
        if self.is_colab:
            try:
                from google.colab import drive
                drive.mount(mount_point)
                self.drive_mounted = True
                print("✅ Google Drive mounted successfully.")
            except ImportError:
                print("⚠️ google.colab not found. Skipping drive mount.")
        else:
            print("ℹ️ Running locally. Skipping drive mount.")

    def resolve_path(self, relative_path: str) -> Path:
        """
        Resolves a relative path to the project root (Current Working Directory).
        Works seamlessly in both local VS Code and Colab (since Bootloader runs os.chdir).
        
        Args:
            relative_path: Path relative to project root (e.g., 'my_models/v7_exp')
        """
        base = Path(os.getcwd())
        final_path = base / relative_path
        final_path.mkdir(parents=True, exist_ok=True)
        return final_path.resolve()

    def __repr__(self):
        env = "Colab" if self.is_colab else "Local"
        return f"<CloudStorage Environment: {env}>"
