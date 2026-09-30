# triage-graph

Work in progress; the full README comes with the CLI and CI. See
[docs/DECISIONS.md](docs/DECISIONS.md) for what this was built against and why.

## Limitations

- **Actions only ever target the alerting service.** Neither the model nor a human
  editing a proposal can act on another service. A proposal to restart a failing
  dependency is refused and becomes `page_human`, and an edit changes the action
  only. This keeps the blast radius small, but it means a correct fix that belongs
  to another service always needs a person.
- **The approval gate trusts the checkpoint database.** Anyone who can write to the
  SQLite file can forge a decision. Decisions are not signed.
