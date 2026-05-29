import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docs: [
    {
      type: "category",
      label: "Getting started",
      collapsed: false,
      items: [
        "getting-started/installation",
        "getting-started/quickstart",
        "getting-started/exposure-modes",
        "getting-started/difficulty-profiles",
      ],
    },
    {
      type: "category",
      label: "Concepts",
      items: [
        "concepts/overview",
        "concepts/world-simulation",
        "concepts/motif-families",
        "concepts/output-bundle",
      ],
    },
    {
      type: "category",
      label: "Reference",
      items: [
        "reference/cli",
        "reference/api",
        "reference/output-bundle",
      ],
    },
    {
      type: "category",
      label: "Dataset — v1",
      items: [
        "dataset/generation-method",
        "dataset/features",
        "dataset/break-me",
        "dataset/acceptance-gates",
        "dataset/v2-decision-log",
        "dataset/release-notes",
      ],
    },
  ],
};

export default sidebars;
