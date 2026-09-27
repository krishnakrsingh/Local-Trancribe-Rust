; VaaniDesk installer (Inno Setup 6).
;
; Owned by A12 (installer/**). Per-user install, no admin required:
;   - PrivilegesRequired=lowest  -> HKCU registry, {localappdata} paths only.
;   - DefaultDirName {localappdata}\VaaniDesk (overrideable with /DIR=).
;   - Add/Remove Programs entry is automatic (Uninstallable=yes, HKCU uninstall key).
;
; ---------------------------------------------------------------------------
; LAYOUT. Load-bearing. Two rules, and both were arrived at the hard way:
;
;   1. `models\` sits DIRECTLY BESIDE the executables, i.e. {app}\models\<dir>.
;      `vaani_models::paths::default_bundled_dir` resolves the bundled root to
;      <exe dir>\models before anything else, so the installed location is the
;      first directory searched. Putting it anywhere else means a first-run user
;      is told to download a model that is already in the next folder.
;
;   2. `runtime\*.dll` are FLAT inside runtime\. cuDNN 9's cudnn64_9.dll is a
;      dispatcher that LoadLibrary's its ten siblings by bare name, and Windows
;      only searches beside the executable. Nest them and GPU acceleration
;      fails *silently* -- the app falls back to the CPU with no error, which
;      reads as "the GPU does not work" rather than "you nested the files".
;      `vaani_inference::RuntimeLocation::discover` finds <exe>\runtime first.
;
; ---------------------------------------------------------------------------
; File associations: one ProgId per kind (VaaniDesk.Audio / VaaniDesk.Video),
; registered under HKCU\Software\Classes from vaani_core::SUPPORTED_EXTENSIONS.
; "Open with VaaniDesk" verbs launch vaani.exe "%1".
;
; Build:  ISCC.exe /DVersion=0.1.0 /DStageDir=... installer\vaani.iss
; (see scripts/release/make-release.ps1, which passes both defines).
#ifndef Version
  #define Version "0.1.0"
#endif
#ifndef StageDir
  #define StageDir "stage"
#endif

[Setup]
AppName=VaaniDesk
AppVersion={#Version}
AppPublisher=VaaniDesk
AppPublisherURL=https://github.com/
DefaultDirName={localappdata}\VaaniDesk
DefaultGroupName=VaaniDesk
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 11 x64 product (MSVC host, x86_64-pc-windows-msvc); declare the floor
; explicitly rather than inheriting a default (ADR-008).
MinVersion=10.0.22000
OutputBaseFilename=VaaniDesk-Setup-{#Version}-win-x64
OutputDir=.
; The distribution is ~1 GB mostly because the speech model is 887 MiB of
; already-compressed fp16 weights. Solid LZMA2 spends a lot of time for very
; little: it is why the CPU pack takes minutes. `fastlzma2` is the right trade
; at this size, and the installer still ends up smaller than the ZIP because
; it stores one copy rather than a directory entry per file.
Compression=lzma2/fast
SolidCompression=yes
LZMANumBlockThreads=4
UninstallDisplayName=VaaniDesk {#Version}
WizardStyle=modern
DisableProgramGroupPage=yes
; The third-party notice is shown in the wizard, not just installed: a user
; redistributing this build should not have to go looking for the licence.
LicenseFile={#StageDir}\licenses\NOTICE.md

[Files]
; --- binaries (built by scripts/release/make-release.ps1) ---
Source: "{#StageDir}\vaani.exe";    DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\vaani-cli.exe"; DestDir: "{app}"; Flags: ignoreversion

; --- ONNX Runtime, FLAT in runtime\ (see header comment) ---
Source: "{#StageDir}\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs

; --- the speech recognition model, 887 MiB, beside the executables ---
; SraVaani 1.0 is MIT-licensed and is bundled deliberately: the product is
; 100% offline and needs no token, no account and no network on first launch.
; The full notice is in licenses\SraVaani-MIT-LICENSE.txt and is installed by
; the `licenses\*` line below.
Source: "{#StageDir}\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs

; --- third-party notices and the model licence, installed not just referenced ---
Source: "{#StageDir}\licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion

; --- documentation, the checksum manifest, and the self-test ---
Source: "{#StageDir}\OFFLINE-INSTALL.md";    DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\vaani-dist.sha256";     DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\vaani-dist.json";       DestDir: "{app}"; Flags: ignoreversion
; The model registry, so the self-test can hash the 887 MiB against the sizes
; and SHA-256s that only exist in this file. Without it `verify-install.ps1` on
; a user's machine can confirm the weights are present but not that they are the
; right ones -- and a silently-corrupt encoder is the right size and the wrong
; bytes, which is exactly the case it exists to catch.
Source: "{#StageDir}\vaani-model-registry.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\verify-install.ps1";    DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; Keep models\ in Add/Remove Programs so a user can see how much disk the
; shipped model occupies, and so it is not silently swept up by "empty folder"
; clean-up utilities.
Name: "{app}\models"; Tasks: none

[Icons]
Name: "{group}\VaaniDesk"; Filename: "{app}\vaani.exe"
Name: "{autodesktop}\VaaniDesk"; Filename: "{app}\vaani.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
; Check the install on completion. Off by default because it hashes 887 MiB and
; takes most of a minute on a slow disk, and an install wizard that appears to
; hang is worse than one that does not.
Name: verifyinstall; Description: "Check the installation after setup (&verify model and runtime)"; GroupDescription: "After install:"; Flags: unchecked

[Run]
; Checked, not blocking, and off by default (see the task above). A missing or
; damaged model is a fact the user can act on -- reinstall -- not a setup
; failure, and failing the install here would leave them with nothing.
;
; Only the self-test runs. `vaani-cli doctor` was previously run here too, which
; duplicated most of the same work and produced a second, less specific report
; nobody could act on.
Filename: "{app}\verify-install.ps1"; Parameters: "-ExpectRuntimeFlavour cpu"; Flags: runhidden; Tasks: verifyinstall; StatusMsg: "Checking the installation..."

[Code]
// Tell Explorer the associations changed so icons/verbs appear without a logoff.
const
  SHCNE_ASSOCCHANGED = $08000000;
  SHCNF_IDLIST = $0000;
procedure SHChangeNotify(wEventID: Integer; uFlags: Cardinal; dwItem1: Longint; dwItem2: Longint);
  external 'SHChangeNotify@shell32.dll stdcall';
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, 0, 0);
end;

[Registry]
; --- VaaniDesk.Audio ProgId (HKCU only: no admin) ---
Root: HKCU; Subkey: "Software\Classes\VaaniDesk.Audio"; ValueType: string; ValueData: "VaaniDesk audio"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\VaaniDesk.Audio\shell\open\command"; ValueType: string; ValueData: """{app}\vaani.exe"" ""%1"""
; --- VaaniDesk.Video ProgId ---
Root: HKCU; Subkey: "Software\Classes\VaaniDesk.Video"; ValueType: string; ValueData: "VaaniDesk video"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\VaaniDesk.Video\shell\open\command"; ValueType: string; ValueData: """{app}\vaani.exe"" ""%1"""
; --- Audio extensions -> VaaniDesk.Audio (from vaani_core::SUPPORTED_EXTENSIONS) ---
Root: HKCU; Subkey: "Software\Classes\.wav"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.mp3"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.m4a"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.aac"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.flac"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.ogg"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.opus"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.wma"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.aiff"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.amr"; ValueType: string; ValueData: "VaaniDesk.Audio"; Flags: uninsdeletevalue
; --- Video extensions -> VaaniDesk.Video ---
Root: HKCU; Subkey: "Software\Classes\.mp4"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.mkv"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.mov"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.webm"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.avi"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.m4v"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.mpg"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.mpeg"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.wmv"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.flv"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.ts"; ValueType: string; ValueData: "VaaniDesk.Video"; Flags: uninsdeletevalue
