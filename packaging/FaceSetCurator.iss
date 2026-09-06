; Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

#define AppName "FaceSet Curator"
#define AppVersion "0.1.0"
#define AppFileVersion "0.1.0.0"
#define AppExeName "FaceSetCurator.exe"

[Setup]
AppId={{1B00AF6C-745B-45A6-A8E6-35E61B73B7B8}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Hanafi Mohd Radi
AppPublisherURL=https://github.com/alberthanafi/FaceSet-Curator-FSC
DefaultDirName={autopf}\FaceSet Curator
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=FaceSetCurator-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
VersionInfoCopyright=Copyright © 2026 Hanafi Mohd Radi. All rights reserved.
VersionInfoVersion={#AppFileVersion}
UninstallDisplayIcon={app}\{#AppExeName}

[Files]
Source: "..\dist\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    if not Exec(ExpandConstant('{app}\{#AppExeName}'), '--doctor', '', SW_HIDE,
      ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
      MsgBox('FaceSet Curator was installed, but CUDA verification failed. The application will not silently use CPU. Open Diagnostics or run fsc doctor --device cuda after repairing the NVIDIA/ONNX runtime.',
        mbError, MB_OK);
  end;
end;
