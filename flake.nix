{
  description = "PyTorch Transformer Sequencer";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
          config.allowUnfree = true;
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            python312
            uv
            gcc13
            cudatoolkit
            cudaPackages.cudnn
            cudaPackages.cuda_cudart
          ];
          shellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
                pkgs.gcc13.cc.lib
                pkgs.cudatoolkit
                pkgs.cudaPackages.cudnn
              ]
            }:$LD_LIBRARY_PATH
            if [ ! -d .venv ]; then
              uv venv .venv --python python3.12
            fi
            source .venv/bin/activate
            echo "Transformer Sequencer dev shell"
            echo "Python: $(python --version)"
          '';
        };
      }
    );
}
