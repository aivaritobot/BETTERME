' BetterMe - Lanzador para Windows (doble clic, SIN terminal)
' Ejecuta la app con pythonw.exe para que no se abra ninguna ventana de consola.

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Carpeta donde está este .vbs
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

' Intentar con pythonw.exe (sin consola). Si no existe, usar python.exe oculto.
cmd = "pythonw.exe -m app.main"

' 0 = ventana oculta, False = no esperar
On Error Resume Next
WshShell.Run cmd, 0, False
If Err.Number <> 0 Then
    Err.Clear
    WshShell.Run "python.exe -m app.main", 0, False
    If Err.Number <> 0 Then
        MsgBox "No se encontró Python. Instala Python 3 desde https://www.python.org/downloads/ y marca 'Add to PATH'.", 16, "BetterMe"
    End If
End If
