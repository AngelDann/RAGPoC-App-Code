; Packages the PyInstaller onedir build (dist\RAGPoC\RAGPoC.exe + _internal\) produced by
; ragpoc.spec into a proper Windows installer. This exists alongside the plain zip that
; release.yml has always published (kept for the in-app updater, see ragpoc/updater.py) --
; the zip is what breaks for first-time installs: Explorer/browsers tag every file extracted
; from a downloaded zip with the NTFS "Mark of the Web", which makes pythonnet's CLR loading
; fail and pywebview silently fall back to opening a browser tab instead of the native window
; (see ragpoc/updater.unblock_downloaded_install, which patches that for the zip path). An
; installer's file-copy step never sets that tag in the first place, so this route sidesteps
; the whole bug class rather than working around it.
;
; Installs per-user under %LOCALAPPDATA% (PrivilegesRequired=lowest, no UAC prompt) rather
; than Program Files: the app treats its own install folder as writable -- data\, .env and
; ragpoc.log all live next to RAGPoC.exe (see BASE_DIR in src/ragpoc/config.py), and the
; in-app self-updater replaces RAGPoC.exe in place -- neither works without elevation under
; Program Files' default ACLs.
;
; Build locally: iscc installer.iss  (uses MyAppVersion "0.0.0-dev" below)
; Build in CI:    iscc /DMyAppVersion="1.2.3" installer.iss  (see .github/workflows/release.yml)

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif

#define MyAppName "RAGPoC"
#define MyAppPublisher "RAGPoC"
#define MyAppExeName "RAGPoC.exe"

[Setup]
; Fixed GUID so successive installer runs are recognized as upgrades of the same app rather
; than separate installs -- never regenerate this.
AppId={{F7526B51-2ACF-4072-86E8-72D298E670B1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=dist\installer
OutputBaseFilename=RAGPoC-Setup
SetupIconFile=assets\ragpoc.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Everything PyInstaller produced for the onedir build -- RAGPoC.exe plus _internal\. Not
; listing data\, .env or ragpoc.log here on purpose: those are created by the app itself at
; first run (see desktop_launcher.py / ensure_directories), not shipped by the installer, so
; an upgrade install never touches or overwrites a user's existing database or config.
Source: "dist\RAGPoC\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
