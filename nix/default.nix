# Copied from https://nixos.org/nixos/nix-pills/callpackage-design-pattern.html
let
  nixpkgs = import <nixpkgs> {};
  allPkgs = nixpkgs // pkgs;
  callPackage = path: overrides:
    let f = import path;
    in f ((builtins.intersectAttrs (builtins.functionArgs f) allPkgs) // overrides);
  packageOverrides = with nixpkgs.python3Packages; callPackage ../python-packages.nix { };
  python = nixpkgs.python3.override {
    packageOverrides = packageOverrides;
  };
  py = python.withPackages (ps: [ ps.pytest ps.pyyaml ps.fv3config ]);
  pkgs = with nixpkgs; {
    fms = callPackage ./fms { };
    esmf = callPackage ./esmf { };
    nceplibs = callPackage ./nceplibs { };
    fv3 = callPackage ./fv3 { };
    python = py;
  };
in pkgs