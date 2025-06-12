{
  description = "A development environment for a PyTorch package.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        pythonEnv = pkgs.python311;

        music21-py = pkgs.python3Packages.buildPythonPackage rec {
          pname = "music21";
          version = "8.1.0";
          pyproject = true;

          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-dVEv0hw4rMfOWerl77nWnH2NHnENFQf5lFKF5EPV1QU";
          };

          nativeBuildInputs = [
            pkgs.python3Packages.chardet
            pkgs.python3Packages.hatchling
            pkgs.python3Packages.joblib
            pkgs.python3Packages.jsonpickle
            pkgs.python3Packages.matplotlib
            pkgs.python3Packages.more-itertools
            pkgs.python3Packages.numpy
            pkgs.python3Packages.requests
            pkgs.python3Packages.setuptools
            pkgs.python3Packages.webcolors
            pkgs.python3Packages.wheel
          ];
        };
      in
      {
        devShell = pkgs.mkShell {
          name = "pytorch-dev-shell";

          buildInputs = with pkgs; [
            pythonEnv
            (pythonEnv.withPackages (
              pyPkgs: with pyPkgs; [
                chardet
                hatchling
                joblib
                jsonpickle
                matplotlib
                more-itertools
                music21-py
                numpy
                poetry-core
                pytorch
                requests
                setuptools
                webcolors
                wheel
              ]
            ))

            # Essential development tools
            git
            bashInteractive
            bash-completion
            openssl
          ];

          shellHook = ''
            echo "--------------------------------------------------------"
            echo "Welcome to the PyTorch development shell!"
            echo "Python version: $(python --version)"
            echo "Poetry version: $(poetry --version)"
            echo "PyTorch version (if installed correctly):"
            python -c "import torch; print(torch.__version__); print(f'CUDA available: {torch.cuda.is_available()}')"
            echo "music21 version (if installed correctly):"
            python -c "import music21; print(music21.__version__)"
            echo "--------------------------------------------------------"

            # Optional: Activate your poetry environment if you have a pyproject.toml
            # cd /path/to/your/pytorch/package # Change to your project directory
            # poetry install # Install dependencies from pyproject.toml
            # poetry shell # Activate the poetry shell for the project
          '';
        };
      }
    );
}
