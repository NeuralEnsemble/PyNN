"""
Generate CITATION.cff from codemeta.json.

codemeta.json is the source of the software metadata; CITATION.cff is generated from it,
so that GitHub can show "Cite this repository". Run from the root of the repository:

    python tools/codemeta2cff.py          # write CITATION.cff
    python tools/codemeta2cff.py --check  # exit with an error if CITATION.cff is out of date

Only the standard library is used. The YAML is written by hand: all strings are written
as JSON strings, which are valid YAML double-quoted scalars.

:copyright: Copyright 2006-2026 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import argparse
import json
import sys
from pathlib import Path

CODEMETA = Path("codemeta.json")
CITATION = Path("CITATION.cff")
HEADER = "# This file is generated from codemeta.json by tools/codemeta2cff.py. Do not edit it.\n"


def orcid(person):
    pid = person.get("@id", "")
    if "orcid.org/" in pid:
        return "https://orcid.org/" + pid.split("orcid.org/")[1]
    return None


def cff_person(person):
    entry = {"given-names": person["givenName"], "family-names": person["familyName"]}
    if orcid(person):
        entry["orcid"] = orcid(person)
    return entry


def doi(identifier):
    return identifier.split("doi.org/")[1]


def spdx_id(license_url):
    return license_url.rstrip("/").split("/")[-1]


def build_citation(codemeta):
    rrid = codemeta["identifier"]
    article = codemeta["referencePublication"]
    year, month, _ = article["datePublished"].split("-")
    volume = article["isPartOf"]
    cff = {
        "cff-version": "1.2.0",
        "message": (f"If you use {codemeta['name']} in your work, please cite the article given "
                    f"in preferred-citation, and give its Research Resource Identifier ({rrid})."),
        "type": "software",
        "title": codemeta["name"],
        "abstract": codemeta["description"].split("\n\n")[0],
        "authors": [cff_person(p) for p in codemeta["author"]],
        "version": codemeta["version"],
        "license": spdx_id(codemeta["license"]),
        "repository-code": codemeta["codeRepository"],
        "identifiers": [
            {"type": "other", "value": rrid, "description": "Research Resource Identifier"},
        ],
        "keywords": codemeta["keywords"],
        "preferred-citation": {
            "type": "article",
            "title": article["name"],
            "authors": [cff_person(p) for p in article["author"]],
            "journal": volume["isPartOf"]["name"],
            "volume": int(volume["volumeNumber"]),
            "start": article["pagination"],
            "year": int(year),
            "month": int(month),
            "doi": doi(article["@id"]),
        },
    }
    if codemeta.get("datePublished"):
        cff["date-released"] = codemeta["datePublished"]
    return cff


def scalar(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"Unsupported value: {value!r}")
    if isinstance(value, int):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def to_yaml(obj, indent=0):
    pad = " " * indent
    lines = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(to_yaml(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{pad}{key}:")
                lines.extend(to_yaml(value, indent))
            else:
                lines.append(f"{pad}{key}: {scalar(value)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                item_lines = to_yaml(item, indent + 2)
                lines.append(f"{pad}- {item_lines[0].lstrip()}")
                lines.extend(item_lines[1:])
            else:
                lines.append(f"{pad}- {scalar(item)}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="check that CITATION.cff is up to date, rather than writing it")
    args = parser.parse_args()
    codemeta = json.loads(CODEMETA.read_text(encoding="utf-8"))
    text = HEADER + "\n".join(to_yaml(build_citation(codemeta))) + "\n"
    if args.check:
        if not CITATION.exists() or CITATION.read_text(encoding="utf-8") != text:
            sys.exit(f"{CITATION} is out of date: run 'python tools/codemeta2cff.py' "
                     f"after changing {CODEMETA}")
        print(f"{CITATION} is up to date")
    else:
        CITATION.write_text(text, encoding="utf-8")
        print(f"Wrote {CITATION}")


if __name__ == "__main__":
    main()
