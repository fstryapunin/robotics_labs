@echo off
REM Set the folder to be zipped
set folderPath=.\src

REM Set the name of the zip archive
set zipFolder=.\brute
set zipName=brute

REM Use PowerShell to zip the folder
powershell Compress-Archive -Path "%folderPath%" -DestinationPath "%zipFolder%"/"%zipName%" -Force

echo Folder has been added to zip archive.
echo Opening in explorer...

powershell explorer "%zipFolder%"