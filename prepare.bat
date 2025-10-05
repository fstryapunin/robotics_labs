@echo off
REM Set the folder to be zipped
set folderPath=.\src

REM Set the name of the zip archive
set zipFile=.\brute\brute

REM Use PowerShell to zip the folder
powershell Compress-Archive -Path "%folderPath%" -DestinationPath "%zipFile%" -Force

echo Folder has been added to zip archive.
pause