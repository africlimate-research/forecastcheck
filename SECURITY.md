# Security

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately, either
through GitHub's [private vulnerability reporting](../../security/advisories/new) on this
repository, or by email to [research@africlimate.ai](mailto:research@africlimate.ai).

Say what you found, how to reproduce it, and what it could affect. We will acknowledge the
report, keep you informed while we fix it, and credit you if you wish.

## Credentials and data

ForecastCheck never needs credentials in the code. Data is read with a key referenced by
its **path** (for example `GOOGLE_APPLICATION_CREDENTIALS`); never paste a key's contents
into a file, a commit, an issue or a chat. If a credential is ever committed by mistake,
report it as above so it can be revoked — deleting the commit is not enough.
