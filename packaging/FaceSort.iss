; Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

#define AppName "FaceSort"
#define AppVersion "0.2.0"
#define AppFileVersion "0.2.0.0"
#define AppExeName "FaceSort.exe"

[Setup]
AppId={{1B00AF6C-745B-45A6-A8E6-35E61B73B7B8}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=HMR Studio
AppPublisherURL=https://github.com/alberthanafi/FaceSort
DefaultDirName={autopf}\FaceSort
DefaultGroupName={#AppName}
UsePreviousGroup=no
OutputDir=..\dist
OutputBaseFilename=FaceSort-Setup
SetupIconFile=..\src\faceset_curator\assets\fsc.ico
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

[InstallDelete]
Type: files; Name: "{app}\FaceSetCurator.exe"
Type: files; Name: "{autodesktop}\FaceSet Curator.lnk"
Type: files; Name: "{autoprograms}\FaceSet Curator\FaceSet Curator.lnk"

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
      MsgBox('FaceSort was installed, but CUDA verification failed. The application will not silently use CPU. Open Diagnostics or run fsc doctor --device cuda after repairing the NVIDIA/ONNX runtime.',
        mbError, MB_OK);
  end;
end;
