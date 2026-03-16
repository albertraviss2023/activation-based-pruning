import os
import sys
from pathlib import Path
from typing import Optional

class CloudStorage:
    """Handles seamless path resolution between Local (VS Code) and Google Colab environments.

    Ensures that file paths are correctly resolved regardless of the execution
    environment, particularly addressing the differences between local file
    systems and the Colab runtime environment.

    Attributes:
        project_name (str): The name of the project, used for identifying the root.
        is_colab (bool): Flag indicating if the script is running in Google Colab.
        drive_mounted (bool): Flag indicating if Google Drive is currently mounted.
    """
    
    def __init__(self, project_name: str = "reducnn"):
        """Initializes CloudStorage with project information.

        Args:
            project_name (str): The name of the project. Defaults to "reducnn".
        """
        self.project_name = project_name
        # Detect Colab by checking if 'google.colab' is loaded in sys.modules
        self.is_colab = 'google.colab' in sys.modules
        self.drive_mounted = False

    def mount_drive(self, mount_point: str = "/content/drive") -> None:
        """Mounts Google Drive if running in a Google Colab environment.

        Args:
            mount_point (str): The path where Google Drive will be mounted.
                Defaults to "/content/drive".

        Returns:
            None
        """
        if self.is_colab:
            try:
                # Use the Colab-specific library to mount Drive
                from google.colab import drive
                drive.mount(mount_point)
                self.drive_mounted = True
                print("✅ Google Drive mounted successfully.")
            except ImportError:
                print("⚠️ google.colab not found. Skipping drive mount.")
        else:
            print("ℹ️ Running locally. Skipping drive mount.")

    def resolve_path(self, relative_path: str) -> Path:
        """Resolves a relative path to the project root (Current Working Directory).

        Works seamlessly in both local VS Code and Colab (since Bootloader runs os.chdir).
        It also creates the directory if it does not already exist.
        
        Args:
            relative_path (str): Path relative to project root (e.g., 'my_models/v7_exp').

        Returns:
            Path: The fully resolved absolute Path object.
        """
        # Start from the current working directory which is assumed to be project root
        base = Path(os.getcwd())
        final_path = base / relative_path
        
        # Ensure the target directory exists before returning the path
        final_path.mkdir(parents=True, exist_ok=True)
        return final_path.resolve()

    def __repr__(self) -> str:
        """Returns a string representation of the CloudStorage environment.

        Returns:
            str: Environment status (Colab or Local).
        """
        env = "Colab" if self.is_colab else "Local"
        return f"<CloudStorage Environment: {env}>"
