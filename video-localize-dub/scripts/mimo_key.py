"""Read / store the Xiaomi MiMo API key. The key is never printed, logged, passed as an argument or written to files.

Lookup order: env XIAOMI_MIMO_API_KEY, env MIMO_API_KEY, then the OS credential store under "Codex/XiaomiMiMoAPI":
  Windows  Credential Manager (generic credential, built in)
  macOS / Linux  system keyring via the `keyring` package (Keychain / Secret Service; setup.py installs it)

  py -3.12 mimo_key.py --check      # configured? where? (never shows the key)
  <key on stdin> | py -3.12 mimo_key.py --store   # the AI does this: key piped on stdin (heredoc), never in argv/files
  py -3.12 mimo_key.py --store --from-env         # copy an env-var key into the credential store
  py -3.12 mimo_key.py --store --console          # hidden prompt, only in the user's own terminal
  py -3.12 mimo_key.py --delete                   # remove the stored key
  py -3.12 mimo_key.py --test       # stores nothing; one tiny API call to confirm the key works
"""

import argparse
import ctypes
import getpass
import os
import sys

TARGET_NAME = "Codex/XiaomiMiMoAPI"
KEYRING_USER = "Xiaomi MiMo API"
CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2

if sys.platform == "win32":
    from ctypes import wintypes

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR),
        ]


def _win_read():
    adv = ctypes.WinDLL("Advapi32.dll")
    adv.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                              ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
    adv.CredReadW.restype = wintypes.BOOL
    adv.CredFree.argtypes = [ctypes.c_void_p]
    ptr = ctypes.POINTER(CREDENTIALW)()
    if not adv.CredReadW(TARGET_NAME, CRED_TYPE_GENERIC, 0, ctypes.byref(ptr)):
        return ""
    try:
        c = ptr.contents
        if not c.CredentialBlob or not c.CredentialBlobSize:
            return ""
        raw = ctypes.string_at(c.CredentialBlob, c.CredentialBlobSize)
        # entries written by CredWriteW / cmdkey are UTF-16LE; tolerate UTF-8 too
        try:
            val = raw.decode("utf-16-le")
            if not val.isprintable():
                raise UnicodeDecodeError("x", b"", 0, 1, "")
        except UnicodeDecodeError:
            val = raw.decode("utf-8", "ignore")
        return val.strip()
    finally:
        adv.CredFree(ptr)


def _win_write(value):
    raw = value.encode("utf-16-le")
    blob = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    c = CREDENTIALW()
    c.Type = CRED_TYPE_GENERIC
    c.TargetName = TARGET_NAME
    c.Comment = "Xiaomi MiMo API key (video-localize-dub)"
    c.CredentialBlobSize = len(raw)
    c.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    c.Persist = CRED_PERSIST_LOCAL_MACHINE
    c.UserName = KEYRING_USER
    adv = ctypes.WinDLL("Advapi32.dll")
    adv.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
    adv.CredWriteW.restype = wintypes.BOOL
    if not adv.CredWriteW(ctypes.byref(c), 0):
        raise ctypes.WinError()


def store_name():
    return "Windows Credential Manager" if sys.platform == "win32" else "system keyring"


def read_credential():
    if sys.platform == "win32":
        return _win_read()
    try:
        import keyring
        return (keyring.get_password(TARGET_NAME, KEYRING_USER) or "").strip()
    except Exception:
        return ""


def write_credential(value):
    value = (value or "").strip()
    if not value:
        raise RuntimeError("Empty key - nothing stored.")
    if sys.platform == "win32":
        _win_write(value)
        return
    try:
        import keyring
    except ImportError:
        raise RuntimeError("Needs the keyring package: run setup.py (or pip install keyring).")
    keyring.set_password(TARGET_NAME, KEYRING_USER, value)


def delete_credential():
    if sys.platform == "win32":
        adv = ctypes.WinDLL("Advapi32.dll")
        adv.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        adv.CredDeleteW(TARGET_NAME, CRED_TYPE_GENERIC, 0)
        return
    try:
        import keyring
        keyring.delete_password(TARGET_NAME, KEYRING_USER)
    except Exception:
        pass


def key_source():
    for name in ("XIAOMI_MIMO_API_KEY", "MIMO_API_KEY"):
        if os.environ.get(name, "").strip():
            return name
    if read_credential():
        return f"{store_name()}:{TARGET_NAME}"
    return None


def load_api_key():
    return (os.environ.get("XIAOMI_MIMO_API_KEY") or os.environ.get("MIMO_API_KEY")
            or read_credential() or "").strip()


def read_stdin_key():
    """Key piped on stdin (first non-empty line). The AI uses this: the key never appears in argv, env or a file."""
    if sys.stdin is None or interactive_console():
        return ""
    for line in sys.stdin.read().splitlines():
        if line.strip():
            return line.strip()
    return ""


def looks_like_key(v):
    return len(v) >= 16 and not any(c.isspace() for c in v)


def interactive_console():
    """A real keyboard is attached (Windows reports NUL as a tty, so ask the console API)."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    if sys.platform != "win32":
        return True
    import msvcrt
    mode = wintypes.DWORD()
    try:
        h = msvcrt.get_osfhandle(sys.stdin.fileno())
    except OSError:
        return False
    return bool(ctypes.windll.kernel32.GetConsoleMode(wintypes.HANDLE(h), ctypes.byref(mode)))


def test_key():
    """One very short TTS request. Prints only OK / the error class, never the key."""
    from mimo_client import MimoError, tts
    try:
        tts("Hi.", "Mia")
        print("MiMo key test: OK (TTS answered)")
        return True
    except MimoError as e:
        print(f"MiMo key test: FAILED ({str(e)[:160]})")
    except Exception as e:
        print(f"MiMo key test: FAILED ({type(e).__name__})")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--store", action="store_true", help="store the key read from stdin (pipe) into the credential store")
    ap.add_argument("--from-env", action="store_true",
                    help="with --store: copy XIAOMI_MIMO_API_KEY / MIMO_API_KEY into the credential store")
    ap.add_argument("--console", action="store_true", help="with --store: hidden prompt in the user's own terminal")
    ap.add_argument("--delete", action="store_true", help="remove the stored key")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.delete:
        delete_credential()
        print(f"Removed {TARGET_NAME} from {store_name()} (if it existed)")
        return
    if a.store:
        if a.from_env:
            val = (os.environ.get("XIAOMI_MIMO_API_KEY") or os.environ.get("MIMO_API_KEY") or "").strip()
        elif a.console:
            if not interactive_console():
                sys.exit("--console needs the user's own terminal: ! py -3.12 mimo_key.py --store --console")
            val = getpass.getpass("Paste MiMo API key (input hidden): ").strip()
        else:
            val = read_stdin_key()
        if not val:
            print("No key received - nothing stored. Pipe it on stdin (see SKILL.md Phase 0), or use --from-env.")
            sys.exit(1)
        if not looks_like_key(val):
            print("That does not look like an API key (too short or contains spaces) - nothing stored.")
            sys.exit(1)
        write_credential(val)
        print(f"Saved to {store_name()} as {TARGET_NAME} (key not shown)")
        return
    if a.test:
        sys.exit(0 if test_key() else 1)
    src = key_source()
    if src:
        print("MiMo key: configured (" + src + ")")
    else:
        print("MiMo key: NOT configured (see SKILL.md Phase 0 - the AI stores it with mimo_key.py --store)")
        sys.exit(1)


if __name__ == "__main__":
    main()
