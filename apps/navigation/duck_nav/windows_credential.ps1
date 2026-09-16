# Internal bridge. Read output is a secret: callers must capture it, never print it.
# The Python caller supplies the fixed $operation before this public source.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class PollenCredential {
    private const string Target = "Pollen/Microduck/Gemini";
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct Credential {
        public uint Flags;
        public uint Type;
        public string TargetName;
        public string Comment;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
        public uint CredentialBlobSize;
        public IntPtr CredentialBlob;
        public uint Persist;
        public uint AttributeCount;
        public IntPtr Attributes;
        public string TargetAlias;
        public string UserName;
    }
    [DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode,
        SetLastError = true)]
    private static extern bool CredRead(string target, uint type, uint flags, out IntPtr ptr);
    [DllImport("advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode,
        SetLastError = true)]
    private static extern bool CredWrite(ref Credential credential, uint flags);
    [DllImport("advapi32.dll")]
    private static extern void CredFree(IntPtr ptr);

    public static string Read() {
        IntPtr ptr;
        if (!CredRead(Target, 1, 0, out ptr)) {
            if (Marshal.GetLastWin32Error() == 1168) return null;
            throw new InvalidOperationException("Credential read failed");
        }
        try {
            var credential = (Credential)Marshal.PtrToStructure(ptr, typeof(Credential));
            // UTF-16 permits viewing/updating this generic credential in Windows.
            return Marshal.PtrToStringUni(credential.CredentialBlob,
                                         (int)credential.CredentialBlobSize / 2);
        } finally { CredFree(ptr); }
    }

    public static void Save(string secret) {
        if (String.IsNullOrWhiteSpace(secret) || secret.Length > 1024)
            throw new ArgumentException("Invalid API key");
        byte[] bytes = Encoding.Unicode.GetBytes(secret);
        IntPtr blob = Marshal.AllocHGlobal(bytes.Length);
        try {
            Marshal.Copy(bytes, 0, blob, bytes.Length);
            var credential = new Credential {
                Type = 1, TargetName = Target, UserName = "GEMINI_API_KEY",
                Comment = "Microduck visual-agent Gemini API key",
                CredentialBlobSize = (uint)bytes.Length, CredentialBlob = blob,
                Persist = 2 // Current Windows user, this computer, across logons.
            };
            if (!CredWrite(ref credential, 0))
                throw new InvalidOperationException("Credential save failed");
        } finally {
            Array.Clear(bytes, 0, bytes.Length);
            Marshal.Copy(bytes, 0, blob, bytes.Length);
            Marshal.FreeHGlobal(blob);
        }
    }
}
'@
    if ($operation -eq 'save') {
        $secret = [Console]::In.ReadToEnd().Trim()
        [PollenCredential]::Save($secret)
        $secret = $null
    } elseif ($operation -eq 'read') {
        $secret = [PollenCredential]::Read()
        if ($null -eq $secret) { exit 3 }
        [Console]::Write($secret)
        $secret = $null
    } else { exit 1 }
    exit 0
} catch {
    # Never emit an exception containing input, script state or a credential blob.
    [Console]::Error.WriteLine('Windows credential operation failed')
    exit 1
}
