Bundled signing key (so the GitHub build works with zero setup)

  file     : invisibility-release.keystore  (PKCS12)
  alias    : invisibility
  password : invisibility-ar   (store and key)

The key is only meant for installing/updating the app on your own phone.
To publish in an app store with a private key, add the repository secrets
KEYSTORE_BASE64, KEYSTORE_PASSWORD, KEY_ALIAS and KEY_PASSWORD - the workflow then
uses them instead of this file (see README.md).
