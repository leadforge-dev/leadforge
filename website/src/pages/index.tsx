import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import CodeBlock from "@theme/CodeBlock";
import clsx from "clsx";

/* ─── Feature cards ──────────────────────────────────────────────────────────── */
const FEATURES = [
  {
    icon: "🌐",
    title: "Simulated commercial worlds",
    body: "Data isn't sampled from a distribution — it emerges from a simulated company, product, buyers, and go-to-market motion, making every dataset narratively consistent.",
  },
  {
    icon: "🔀",
    title: "Variable hidden DGP",
    body: "Five motif families (fit-dominant, intent-dominant, sales-execution-sensitive, demo/trial-mediated, buying-committee-friction) are stochastically rewired so no two datasets share the same causal structure.",
  },
  {
    icon: "📐",
    title: "Three difficulty tiers",
    body: "Intro, intermediate, and advanced — calibrated by signal-to-noise ratio and conversion rate so you can benchmark a novice project, a serious model, or a stress-test in the same framework.",
  },
  {
    icon: "🔍",
    title: "Full truth for instructors",
    body: "The instructor companion ships the hidden causal graph, latent registry, mechanism summary, and full-horizon relational tables — everything redacted from the student view.",
  },
  {
    icon: "🔗",
    title: "9-table relational output",
    body: "Accounts, contacts, leads, touches, sessions, sales activities, opportunities, customers, and subscriptions — all with deterministic IDs and FK integrity.",
  },
  {
    icon: "🔒",
    title: "Leakage-free by construction",
    body: "Every public feature is snapshot-safe: no post-anchor events, no terminal-stage columns, no conversion-conditional tables. The redaction contract is code, not convention.",
  },
];

/* ─── Tier rows ──────────────────────────────────────────────────────────────── */
const TIERS = [
  {
    badge: "intro",
    badgeCls: "tier-row__badge--intro",
    desc: "Strong signal, low noise. Good for first-time learners and sanity-checking pipelines.",
    metrics: "AUC ≈ 0.89 · ~28% conversion",
  },
  {
    badge: "intermediate",
    badgeCls: "tier-row__badge--inter",
    desc: "Realistic noise, moderate signal. The canonical benchmark tier.",
    metrics: "AUC ≈ 0.79 · ~18% conversion",
  },
  {
    badge: "advanced",
    badgeCls: "tier-row__badge--advanced",
    desc: "High noise, weak signal, rare positive class. Challenges experienced practitioners.",
    metrics: "AUC ≈ 0.68 · ~8% conversion",
  },
];

/* ─── Code snippets ──────────────────────────────────────────────────────────── */
const CLI_SNIPPET = `# Generate a full bundle
leadforge generate \\
  --recipe b2b_saas_procurement_v1 \\
  --seed 42 --mode student_public \\
  --difficulty intermediate \\
  --n-leads 5000 --out ./out/bundle

# Inspect & validate
leadforge inspect ./out/bundle
leadforge validate ./out/bundle`;

const API_SNIPPET = `from leadforge.api import Generator

gen = Generator.from_recipe(
    "b2b_saas_procurement_v1",
    seed=42,
    exposure_mode="student_public",
)
bundle = gen.generate(
    n_leads=5000,
    difficulty="intermediate",
)
bundle.save("./out/bundle")`;

/* ─── Component ──────────────────────────────────────────────────────────────── */
export default function Home(): React.ReactElement {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}
    >
      {/* ── Hero ─────────────────────────────────────────────────────────────── */}
      <header className="hero--leadforge">
        <h1 className="hero__title">leadforge</h1>
        <p className="hero__subtitle">
          Narrative-grounded synthetic CRM datasets generated from simulated
          commercial worlds — for teaching, benchmarks, and research.
        </p>
        <div className="hero__buttons">
          <Link className="button button--primary button--lg" to="/docs/getting-started/installation">
            Get started →
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1"
          >
            Browse on HuggingFace
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="https://github.com/leadforge-dev/leadforge"
          >
            GitHub
          </Link>
        </div>
        <div className="install-pill">
          <span className="install-pill__label">install</span>
          pip install leadforge
        </div>
      </header>

      {/* ── Feature grid ──────────────────────────────────────────────────────── */}
      <section className="features">
        <h2 className="section-heading">Why leadforge?</h2>
        <p className="section-sub">
          Public lead-scoring datasets are too small, too overused, or too shallow to sustain
          serious teaching or research.{" "}
          <strong>leadforge</strong> generates datasets that feel like they came from a real CRM.
        </p>
        <div className="features__grid">
          {FEATURES.map((f) => (
            <div key={f.title} className="feature-card">
              <span className="feature-card__icon">{f.icon}</span>
              <div className="feature-card__title">{f.title}</div>
              <p className="feature-card__body">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Code strip ────────────────────────────────────────────────────────── */}
      <section className="code-strip">
        <div className="code-strip__inner">
          <div>
            <div className="code-strip__label">CLI</div>
            <CodeBlock language="bash">{CLI_SNIPPET}</CodeBlock>
          </div>
          <div>
            <div className="code-strip__label">Python API</div>
            <CodeBlock language="python">{API_SNIPPET}</CodeBlock>
          </div>
        </div>
      </section>

      {/* ── Tiers ────────────────────────────────────────────────────────────── */}
      <section className="tiers">
        <div className="tiers__inner">
          <h2 className="tiers__heading">Three difficulty tiers, one dataset family</h2>
          <p className="tiers__sub">
            All tiers share the same fictional company and causal structure. Only signal
            strength, noise, and missingness differ.
          </p>
          {TIERS.map((t) => (
            <div key={t.badge} className="tier-row">
              <span className={clsx("tier-row__badge", t.badgeCls)}>{t.badge}</span>
              <span className="tier-row__desc">{t.desc}</span>
              <span className="tier-row__metrics">{t.metrics}</span>
            </div>
          ))}
          <p style={{ textAlign: "center", marginTop: "1.5rem", fontSize: "0.9rem", opacity: 0.7 }}>
            Each tier ships 5,000 leads · 70 / 15 / 15 train/valid/test Parquet splits ·
            9-table relational bundle
          </p>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────────────────────────── */}
      <section
        style={{
          padding: "4rem 2rem",
          textAlign: "center",
          borderTop: "1px solid var(--ifm-toc-border-color)",
        }}
      >
        <h2 style={{ fontSize: "1.6rem", fontWeight: 700, marginBottom: "0.75rem" }}>
          Ready to use it?
        </h2>
        <p style={{ color: "var(--ifm-color-content-secondary)", marginBottom: "2rem" }}>
          Download the v1 dataset on HuggingFace or Kaggle, or generate your own with the
          Python package.
        </p>
        <div className="hero__buttons">
          <Link
            className="button button--primary button--lg"
            to="https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1"
          >
            HuggingFace dataset ↗
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/getting-started/installation"
          >
            Read the docs
          </Link>
        </div>
      </section>
    </Layout>
  );
}
