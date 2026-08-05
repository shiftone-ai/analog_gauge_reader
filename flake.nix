{
  description = "analog_gauge_reader development environment (uv + Python 3.10)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          # mmcv は macOS 向けの wheel が配布されておらず、torch の cpp_extension 経由で
          # C++/Objective-C++ 拡張をソースビルドする。これは Xcode の clang と macOS SDK を
          # 前提にしているため、darwin では nix の cc を PATH に入れない (mkShellNoCC)。
          # Linux では OpenMMLab の manylinux wheel が使えるが、他パッケージのビルド用に cc を残す。
          mkShell' = if pkgs.stdenv.isDarwin then pkgs.mkShellNoCC else pkgs.mkShell;
        in
        {
          default = mkShell' {
            packages = with pkgs; [
              uv
              git
              git-lfs # models/ と dependencies/ のチェックポイントは LFS 管理
              ninja # torch の cpp_extension が mmcv のビルドに使う
              pre-commit
            ];

            env = {
              # Python 本体は uv が管理する (.python-version 参照)。
              # nixpkgs には 3.10 系が無いため、uv の standalone build を使う。
              UV_PYTHON_DOWNLOADS = "automatic";
            };

            # direnv 経由の自動ロード時は静かにする
            shellHook = ''
              if [ -z "''${DIRENV_IN_ENVRC:-}" ]; then
                echo "analog_gauge_reader dev shell"
                echo "  uv sync                            # 依存をインストール (mmcv は初回のみソースビルド)"
                echo "  git lfs install --local            # clone 直後に一度だけ"
                echo "  git lfs pull --include='models/*'  # 重みを取得"
                echo "  uv run python pipeline.py --help"
              fi
            '';
          };
        });
    };
}
