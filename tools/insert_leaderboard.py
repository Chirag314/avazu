import os, csv, sys, datetime

START_MARK = "<!-- START:WANDB-LEADERBOARD -->"
END_MARK = "<!-- END:WANDB-LEADERBOARD -->"

def csv_to_markdown(csv_path, max_rows=10):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows: break
            rows.append(row)
    if not rows:
        return "No runs yet."
    headers = list(rows[0].keys())
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"]*len(headers)) + " |")
    for r in rows:
        md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(md)

def main():
    entity = os.getenv("WANDB_ENTITY", "").strip()
    project = os.getenv("WANDB_PROJECT", "").strip()
    link = f"https://wandb.ai/{entity}/{project}" if entity and project else ""

    lb_csv = "leaderboard/leaderboard.csv"
    if not os.path.exists(lb_csv):
        print("leaderboard.csv not found. Exiting.", file=sys.stderr)
        sys.exit(0)

    md_table = csv_to_markdown(lb_csv, max_rows=10)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    block_lines = []
    block_lines.append(START_MARK)
    block_lines.append("## W&B Leaderboard (auto-generated)")
    if link:
        block_lines.append(f"[Open in W&B]({link})")
    block_lines.append(f"Last update: {now}")
    block_lines.append("")
    block_lines.append(md_table)
    block_lines.append(END_MARK)
    block = "\n".join(block_lines)

    readme_path = "README.md"
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    if START_MARK in content and END_MARK in content:
        start = content.index(START_MARK)
        end = content.index(END_MARK) + len(END_MARK)
        new_content = content[:start] + block + content[end:]
    else:
        # append at end
        new_content = content.rstrip() + "\n\n" + block + "\n"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("README.md updated with W&B Leaderboard.")

if __name__ == "__main__":
    main()
