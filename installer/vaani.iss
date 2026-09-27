; VaaniDesk installer (Inno Setup 6).
;
; Owned by agent A8 (installer/**). Per-user install, no admin required:
;   - PrivilegesRequired=lowest  -> HKCU registry, {localappdata} paths only.
;   - DefaultDirName {localappdata}\VaaniDesk (overrideable with /DIR=).
;   - Add/Remove Programs entry is automatic (Uninstallable=yes, HKCU uninstall key).
;
; Layout (load-bearing, ADR-007/ADR-008): vaani.exe, vaani-cli.exe and
; runtime\*.dll sit with runtime\ as a SUBDIRECTORY of the install root, all
; DLLs FLAT inside it. cuDNN 9's cudnn64_9.dll LoadLibrary's its siblings by
; bare name; nesting them deeper breaks GPU acceleration silently (CPU fallback
; with no error). RuntimeLocation::discover finds <exe>\runtime first.
;
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
Compression=lzma2/max
SolidCompression=yes
UninstallDisplayName=VaaniDesk {#Version}
; No LicenseFile: the repo ships no top-level LICENSE yet (see OPEN ISSUES in
; the A8 report). The runtime redistributable licence stages as runtime\LICENSE.

[Files]
; Binaries (built by scripts/release/make-release.ps1 --release).
Source: "{#StageDir}\vaani.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\vaani-cli.exe"; DestDir: "{app}"; Flags: ignoreversion
; Runtime DLL manifest note: all DLLs FLAT in runtime\. Do not reorganise into
; per-component subdirectories (see header comment).
Source: "{#StageDir}\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs
Source: "{#StageDir}\runtime-manifest.json"; DestDir: "{app}\runtime"; Flags: ignoreversion

[Icons]
Name: "{group}\VaaniDesk"; Filename: "{app}\vaani.exe"
Name: "{autodesktop}\VaaniDesk"; Filename: "{app}\vaani.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

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

[Run]
; First thing after install: the verification surface (ADR-008). Checked, not
; blocking: a missing runtime is a fact the user can act on, not a setup failure.
Filename: "{app}\vaani-cli.exe"; Parameters: "doctor"; Flags: runhidden; StatusMsg: "Checking the installation..."

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
