{pkgs}: {
  deps = [
    pkgs.unzip
    pkgs.wget
    pkgs.glibcLocales
    pkgs.geos
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.nodejs
    pkgs.geckodriver
    pkgs.firefox
    pkgs.arrow-cpp
  ];
}
