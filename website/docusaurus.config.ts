import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "leadforge",
  tagline:
    "Generate narrative-grounded synthetic CRM datasets from simulated commercial worlds.",
  favicon: "img/favicon.ico",

  future: { v4: true },

  url: "https://leadforge-dev.github.io",
  baseUrl: "/leadforge/",

  organizationName: "leadforge-dev",
  projectName: "leadforge",
  trailingSlash: false,

  onBrokenLinks: "warn",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          editUrl:
            "https://github.com/leadforge-dev/leadforge/edit/main/website/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/leadforge-social.png",
    colorMode: {
      defaultMode: "dark",
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "leadforge",
      logo: {
        alt: "leadforge logo",
        src: "img/logo.svg",
        srcDark: "img/logo-dark.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docs",
          position: "left",
          label: "Docs",
        },
        {
          to: "/docs/dataset/generation-method",
          label: "Dataset",
          position: "left",
        },
        {
          href: "https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1",
          label: "HuggingFace ↗",
          position: "right",
        },
        {
          href: "https://www.kaggle.com/datasets/derelictpanda/leadforge-lead-scoring-v1",
          label: "Kaggle ↗",
          position: "right",
        },
        {
          href: "https://github.com/leadforge-dev/leadforge",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Getting started",
          items: [
            { label: "Installation", to: "/docs/getting-started/installation" },
            { label: "Quickstart", to: "/docs/getting-started/quickstart" },
            { label: "Exposure modes", to: "/docs/getting-started/exposure-modes" },
          ],
        },
        {
          title: "Reference",
          items: [
            { label: "CLI", to: "/docs/reference/cli" },
            { label: "Python API", to: "/docs/reference/api" },
            { label: "Output bundle", to: "/docs/reference/output-bundle" },
          ],
        },
        {
          title: "Dataset",
          items: [
            { label: "Generation method", to: "/docs/dataset/generation-method" },
            { label: "Feature dictionary", to: "/docs/dataset/features" },
            { label: "Break-me guide", to: "/docs/dataset/break-me" },
            { label: "v2 decision log", to: "/docs/dataset/v2-decision-log" },
          ],
        },
        {
          title: "Project",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/leadforge-dev/leadforge",
            },
            {
              label: "HuggingFace",
              href: "https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1",
            },
            {
              label: "Kaggle",
              href: "https://www.kaggle.com/datasets/derelictpanda/leadforge-lead-scoring-v1",
            },
            {
              label: "Issues",
              href: "https://github.com/leadforge-dev/leadforge/issues",
            },
          ],
        },
      ],
      copyright: `MIT License · <a href="https://huggingface.co/shaypal5" style="color: var(--ifm-footer-link-color)">Shay Palachy Affek</a>`,
    },
    prism: {
      theme: prismThemes.oneDark,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ["bash", "python", "json", "yaml"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
