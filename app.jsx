import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  CircleHelp,
  Database,
  GraduationCap,
  LayoutDashboard,
  LoaderCircle,
  RefreshCw,
  Settings as SettingsIcon,
  ShieldCheck,
  Sparkles,
  Target,
  Wifi,
  X
} from 'lucide-react';

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'https://stuentlens-ai.onrender.com'
).replace(/\/$/, '');

const OUTCOME_LABELS = {
  0: 'Graduate',
  1: 'Dropout',
  2: 'Enrolled'
};

const FEATURES = [
  {
    name: 'Curricular units 1st sem (credited)',
    key: 'Curricular units 1st sem (credited)',
    section: 'Academic Performance',
    type: 'number',
    min: 0,
    step: 1,
    help: 'Credits recognized from prior study.'
  },
  {
    name: 'Curricular units 1st sem (enrolled)',
    key: 'Curricular units 1st sem (enrolled)',
    section: 'Academic Performance',
    type: 'number',
    min: 0,
    step: 1,
    help: 'Number of first-semester curricular units enrolled.'
  },
  {
    name: 'Curricular units 1st sem (evaluations)',
    key: 'Curricular units 1st sem (evaluations)',
    section: 'Academic Performance',
    type: 'number',
    min: 0,
    step: 1,
    help: 'Number of first-semester evaluations.'
  },
  {
    name: 'Curricular units 1st sem (approved)',
    key: 'Curricular units 1st sem (approved)',
    section: 'Academic Performance',
    type: 'number',
    min: 0,
    step: 1,
    help: 'Number of first-semester units approved.'
  },
  {
    name: 'Curricular units 1st sem (grade)',
    key: 'Curricular units 1st sem (grade)',
    section: 'Academic Performance',
    type: 'number',
    min: 0,
    step: 0.01,
    help: 'Recorded first-semester grade.'
  },
  {
    name: 'Tuition fees up to date',
    key: 'Tuition fees up to date',
    section: 'Financial Information',
    type: 'binary',
    help: 'Are tuition fees currently up to date?'
  },
  {
    name: 'Scholarship holder',
    key: 'Scholarship holder',
    section: 'Financial Information',
    type: 'binary',
    help: 'Does the student hold a scholarship?'
  },
  {
    name: 'Debtor',
    key: 'Debtor',
    section: 'Financial Information',
    type: 'binary',
    help: 'Is the student recorded as a debtor?'
  },
  {
    name: 'Age at enrollment',
    key: 'Age at enrollment',
    section: 'Student Background',
    type: 'number',
    min: 16,
    max: 100,
    step: 1,
    help: 'Student age when enrolling.'
  },
  {
    name: 'Admission grade',
    key: 'Admission grade',
    section: 'Student Background',
    type: 'number',
    min: 0,
    max: 200,
    step: 0.01,
    help: 'Admission grade on the dataset scale (0–200).'
  },
  {
    name: 'Gender',
    key: 'Gender',
    section: 'Student Background',
    type: 'select',
    options: [
      { value: 1, label: 'Male' },
      { value: 0, label: 'Female' }
    ],
    help: 'Select the student gender.'
  },
  {
    name: 'Application mode',
    key: 'Application mode',
    section: 'Enrollment Information',
    type: 'select',
    options: [
      1, 2, 5, 7, 10, 15, 16, 17, 18,
      26, 27, 39, 42, 43, 44, 51, 53, 57
    ].map(value => ({
      value,
      label: String(value)
    })),
    help: 'Choose an application-mode code available in the form.'
  },
  {
    name: 'Course',
    key: 'Course',
    section: 'Enrollment Information',
    type: 'number',
    min: 1,
    step: 1,
    help: 'Enter any positive integer course code.'
  }
];

const SECTIONS = [
  'Academic Performance',
  'Financial Information',
  'Student Background',
  'Enrollment Information'
];

const NAV = [
  { id: 'Overview', icon: LayoutDashboard },
  { id: 'Predict Outcome', icon: Target },
  { id: 'Model Insights', icon: Sparkles },
  { id: 'Settings', icon: SettingsIcon }
];

const blankForm = () =>
  Object.fromEntries(FEATURES.map(feature => [feature.key, '']));

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });

  const body = await response.json().catch(() => ({
    detail: 'The server returned an unreadable response.'
  }));

  if (!response.ok) {
    const message =
      typeof body.detail === 'string'
        ? body.detail
        : body.detail?.message ||
          'The request could not be completed.';

    throw new Error(message);
  }

  return body;
}

function prettyLabel(value) {
  if (value === null || value === undefined || value === '') {
    return 'Unknown';
  }

  const label = String(value);

  return OUTCOME_LABELS[label] ?? `Unknown class (${label})`;
}

function MetricCard({ label, value, icon: Icon, note }) {
  return (
    <article className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <span className="metric-icon">
          <Icon size={18} />
        </span>
      </div>

      <div className="metric-value">{value ?? '—'}</div>
      <div className="metric-note">{note}</div>
    </article>
  );
}

function Empty({ title, detail }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <BookOpen size={22} />
      </div>

      <h3>{title}</h3>
      <p>{detail}</p>
    </div>
  );
}

function StatusPill({ ok, children }) {
  return (
    <span className={`status-pill ${ok ? 'is-ok' : 'is-down'}`}>
      <span className="status-dot" />
      {children}
    </span>
  );
}

export default function App() {
  const [page, setPage] = useState('Overview');

  const [health, setHealth] = useState(null);
  const [info, setInfo] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);

  const [form, setForm] = useState(blankForm);
  const [formError, setFormError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const [result, setResult] = useState(null);

  const refreshSystem = useCallback(async () => {
    setHealthLoading(true);

    try {
      const [healthData, modelInfo] = await Promise.all([
        api('/health'),
        api('/model-info')
      ]);

      setHealth(healthData);
      setInfo(modelInfo);
    } catch (error) {
      setHealth({
        model_loaded: false,
        database_connected: false,
        status: 'offline',
        model_error: error.message
      });

      setInfo(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshSystem();
  }, [refreshSystem]);

  function changePage(id) {
    setPage(id);

    if (id !== 'Predict Outcome') {
      setResult(null);
    }
  }

  function updateField(key, value) {
    setForm(previous => ({
      ...previous,
      [key]: value
    }));

    setFormError('');
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setFormError('');

    const missing = FEATURES.filter(
      feature => form[feature.key] === ''
    );

    if (missing.length > 0) {
      setFormError(
        `Please complete: ${missing
          .map(feature => feature.name)
          .join(', ')}.`
      );
      return;
    }

    const payload = {};

    for (const feature of FEATURES) {
      const rawValue = form[feature.key];
      const value = Number(rawValue);

      if (!Number.isFinite(value)) {
        setFormError(`${feature.name} must be a valid number.`);
        return;
      }

      if (feature.key === 'Course') {
        if (!Number.isInteger(value) || value <= 0) {
          setFormError(
            'Course must be a positive whole-number code.'
          );
          return;
        }
      } else {
        if (
          feature.min !== undefined &&
          value < feature.min
        ) {
          setFormError(
            `${feature.name} must be at least ${feature.min}.`
          );
          return;
        }

        if (
          feature.max !== undefined &&
          value > feature.max
        ) {
          setFormError(
            `${feature.name} must be no more than ${feature.max}.`
          );
          return;
        }
      }

      if (
        feature.step === 1 &&
        !Number.isInteger(value)
      ) {
        setFormError(`${feature.name} must be a whole number.`);
        return;
      }

      payload[feature.key] = value;
    }

    setSubmitting(true);

    try {
      const prediction = await api('/predict', {
        method: 'POST',
        body: JSON.stringify(payload)
      });

      setResult(prediction);

      await refreshSystem();
    } catch (error) {
      setFormError(error.message);
    } finally {
      setSubmitting(false);
    }
  }

  const classNames = info?.class_labels || [];

  const pageTitle =
    page === 'Overview' ? 'Overview Dashboard' : page;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <GraduationCap size={23} />
          </div>

          <div>
            <div className="brand-name">
              StudentLens <span>AI</span>
            </div>

            <div className="brand-tagline">
              See Potential. Predict Success.
            </div>
          </div>
        </div>

        <div className="nav-caption">WORKSPACE</div>

        <nav className="side-nav">
          {NAV.map(({ id, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => changePage(id)}
            >
              <Icon size={18} />
              <span>{id}</span>

              {id === 'Predict Outcome' && (
                <span className="nav-plus">+</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-help">
            <div className="help-icon">
              <CircleHelp size={17} />
            </div>

            <strong>Make informed decisions</strong>

            <p>
              Predictions are estimates to support—not replace—academic
              guidance.
            </p>
          </div>

          <div className="sidebar-footer">
            <span className="footer-dot" />
            <span>StudentLens AI</span>
            <span className="version">v1.0</span>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="mobile-brand">
            <GraduationCap size={22} />
            StudentLens <b>AI</b>
          </div>

          <div className="breadcrumb">
            Workspace <span>/</span> <strong>{pageTitle}</strong>
          </div>

          <div className="topbar-right">
            <span className="topbar-status">
              <span
                className={`tiny-dot ${
                  health?.model_loaded ? '' : 'down'
                }`}
              />

              {health?.model_loaded
                ? 'Model online'
                : 'Model unavailable'}
            </span>

            <div className="avatar">SL</div>
          </div>
        </header>

        <div className="page-content">
          {page === 'Overview' && (
            <section>
              <div className="welcome-row">
                <div>
                  <div className="eyebrow">
                    <Sparkles size={14} />
                    STUDENT PREDICTION PLATFORM
                  </div>

                  <h1>See the bigger picture.</h1>

                  <p className="page-subtitle">
                    Generate an individual student outcome estimate
                    using the existing trained model.
                  </p>
                </div>

                <button
                  className="btn btn-primary"
                  onClick={() => changePage('Predict Outcome')}
                >
                  <Target size={17} />
                  New Prediction
                  <ArrowRight size={16} />
                </button>
              </div>

              <div className="system-banner">
                <div className="system-banner-icon">
                  <Activity size={19} />
                </div>

                <div className="system-banner-copy">
                  <strong>System status</strong>

                  <span>
                    {healthLoading
                      ? 'Checking services…'
                      : health?.model_loaded &&
                          health?.database_connected
                        ? 'Model and database are connected.'
                        : !health?.model_loaded
                          ? 'Model unavailable. Check backend configuration.'
                          : 'Database unavailable. Prediction saving may not work.'}
                  </span>
                </div>

                <div className="system-badges">
                  <StatusPill ok={!!health?.model_loaded}>
                    Model {health?.model_loaded ? 'Ready' : 'Offline'}
                  </StatusPill>

                  <StatusPill ok={!!health?.database_connected}>
                    Database{' '}
                    {health?.database_connected
                      ? 'Connected'
                      : 'Unavailable'}
                  </StatusPill>
                </div>
              </div>

              <div className="metrics-grid">
                <MetricCard
                  label="Model Status"
                  value={
                    healthLoading
                      ? 'Checking'
                      : health?.model_loaded
                        ? 'Ready'
                        : 'Offline'
                  }
                  icon={Activity}
                  note="Existing trained model"
                />

                <MetricCard
                  label="Database"
                  value={
                    health?.database_connected
                      ? 'Connected'
                      : 'Unavailable'
                  }
                  icon={Database}
                  note="Backend database connection"
                />

                <MetricCard
                  label="Model Type"
                  value={info?.model_type || '—'}
                  icon={Target}
                  note="Deployed classifier"
                />

                <MetricCard
                  label="Outcome Classes"
                  value={classNames.length || '—'}
                  icon={GraduationCap}
                  note="Classes reported by the model"
                />
              </div>

              {/* System overview section removed as requested. */}

              <div className="dashboard-grid">
                <article className="panel action-panel">
                  <div className="action-art">
                    <div className="art-circle art-one" />
                    <div className="art-circle art-two" />

                    <div className="art-icon">
                      <Target size={25} />
                    </div>
                  </div>

                  <span className="eyebrow">READY WHEN YOU ARE</span>

                  <h3>Explore a student outcome</h3>

                  <p>
                    Enter academic, financial, and enrollment information
                    to generate a model estimate.
                  </p>

                  <button
                    className="btn btn-primary btn-wide"
                    onClick={() => changePage('Predict Outcome')}
                  >
                    Start a prediction
                    <ArrowRight size={16} />
                  </button>

                  <div className="action-note">
                    <ShieldCheck size={15} />
                    Uses your existing trained model
                  </div>
                </article>
              </div>

              <div className="privacy-note">
                <ShieldCheck size={17} />

                <div>
                  <strong>Privacy-focused dashboard</strong>

                  <span>
                    The dashboard does not show shared prediction
                    history. Predictions are estimates, not guarantees.
                  </span>
                </div>
              </div>
            </section>
          )}

          {page === 'Predict Outcome' && (
            <section>
              <div className="page-heading">
                <div className="eyebrow">
                  <Target size={14} />
                 STUDENT SUCCESS STARTS HERE
                </div>

                <h1>Predict student outcome</h1>

                <p className="page-subtitle">
    
                </p>
              </div>

              {result ? (
                <div className="result-layout">
                  <article className="result-card">
                    <div className="result-success">
                      <CheckCircle2 size={19} />
                      Prediction complete
                    </div>

                    <div className="result-icon">
                      <GraduationCap size={28} />
                    </div>

                    <div className="result-label">
                      PREDICTED STUDENT OUTCOME
                    </div>

                    <h2>{prettyLabel(result.predicted_outcome)}</h2>

                    <p className="result-description">
                      The saved model produced this outcome estimate
                      for the submitted feature values.
                    </p>

                    {result.class_probabilities && (
                      <div className="probability-block">
                        <h3>Outcome probabilities</h3>

                        {Object.entries(result.class_probabilities).map(
                          ([label, value], index) => {
                            const probability = Number(value);

                            const percentage =
                              Number.isFinite(probability)
                                ? Math.min(
                                    100,
                                    Math.max(0, probability * 100)
                                  )
                                : 0;

                            return (
                              <div className="prob-row" key={label}>
                                <div>
                                  <span>{prettyLabel(label)}</span>

                                  <strong>
                                    {Number.isFinite(probability)
                                      ? `${percentage.toFixed(1)}%`
                                      : '—'}
                                  </strong>
                                </div>

                                <div className="progress-track">
                                  <div
                                    className={`progress-fill fill-${
                                      index % 4
                                    }`}
                                    style={{
                                      width: `${percentage}%`
                                    }}
                                  />
                                </div>
                              </div>
                            );
                          }
                        )}
                      </div>
                    )}

                    <div className="disclaimer">
                      <AlertCircle size={16} />
                      This is a model estimate, not a guaranteed
                      academic outcome.
                    </div>

                    <div className="result-actions">
                      <button
                        className="btn btn-primary"
                        onClick={() => {
                          setResult(null);
                          setFormError('');
                        }}
                      >
                        Make another prediction
                      </button>

                      <button
                        className="btn btn-secondary"
                        onClick={() => {
                          setForm(blankForm());
                          setResult(null);
                          setFormError('');
                        }}
                      >
                        Clear form
                      </button>
                    </div>
                  </article>
                </div>
              ) : (
                <form
                  className="prediction-form"
                  onSubmit={submitPrediction}
                  noValidate
                >
                  {SECTIONS.map((section, index) => (
                    <section className="form-section" key={section}>
                      <div className="form-section-heading">
                        <div className="step-number">
                          0{index + 1}
                        </div>

                        <div>
                          <h2>{section}</h2>

                          <p>
                            {
                              [
                                'First-semester academic progress and grades.',
                                'Tuition status and financial support indicators.',
                                'Student age, admission score, and encoded gender.',
                                'Enter enrollment and course codes.'
                              ][index]
                            }
                          </p>
                        </div>
                      </div>

                      <div className="form-grid">
                        {FEATURES.filter(
                          feature => feature.section === section
                        ).map(feature => (
                          <label className="field" key={feature.key}>
                            <span className="field-label">
                              {feature.name}
                              <span className="required">*</span>
                            </span>

                            {feature.type === 'binary' ? (
                              <select
                                value={form[feature.key]}
                                onChange={event =>
                                  updateField(
                                    feature.key,
                                    event.target.value
                                  )
                                }
                              >
                                <option value="">Choose an option</option>
                                <option value="1">Yes</option>
                                <option value="0">No</option>
                              </select>
                            ) : feature.type === 'select' ? (
                              <select
                                value={form[feature.key]}
                                onChange={event =>
                                  updateField(
                                    feature.key,
                                    event.target.value
                                  )
                                }
                              >
                                <option value="">Choose an option</option>

                                {feature.options.map(option => (
                                  <option
                                    key={option.value}
                                    value={option.value}
                                  >
                                    {option.label}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="number"
                                min={feature.min}
                                max={feature.max}
                                step={feature.step}
                                value={form[feature.key]}
                                onChange={event =>
                                  updateField(
                                    feature.key,
                                    event.target.value
                                  )
                                }
                                placeholder={
                                  feature.key === 'Course'
                                    ? 'Enter any course code'
                                    : 'Enter value'
                                }
                              />
                            )}

                            <small>{feature.help}</small>
                          </label>
                        ))}
                      </div>
                    </section>
                  ))}

                  {formError && (
                    <div className="error-box">
                      <AlertCircle size={17} />
                      <span>{formError}</span>

                      <button
                        type="button"
                        onClick={() => setFormError('')}
                        aria-label="Dismiss error"
                      >
                        <X size={15} />
                      </button>
                    </div>
                  )}

                  <div className="form-bottom">
                    <p>
                      <ShieldCheck size={16} />
                      Fields are sent to your configured backend.
                      Predictions are not guarantees.
                    </p>

                    <button
                      className="btn btn-primary btn-submit"
                      type="submit"
                      disabled={submitting}
                    >
                      {submitting ? (
                        <>
                          <LoaderCircle className="spin" size={17} />
                          Predicting…
                        </>
                      ) : (
                        <>
                          Predict Outcome
                          <ArrowRight size={16} />
                        </>
                      )}
                    </button>
                  </div>
                </form>
              )}
            </section>
          )}

          {page === 'Model Insights' && (
            <section>
              <div className="page-heading">
                <div className="eyebrow">
                  <Sparkles size={14} />
                  MODEL TRANSPARENCY
                </div>

                <h1>Model insights</h1>

                <p className="page-subtitle">
                  Verified information about the deployed model artifact.
                </p>
              </div>

              <div className="insight-grid">
                <article className="panel insight-main">
                  <div className="insight-icon">
                    <Activity size={20} />
                  </div>

                  <span className="eyebrow">CLASSIFIER</span>

                  <h2>{info?.model_type || 'SVC'}</h2>

                  <p>
                    Support Vector Classifier within the supplied
                    preprocessing pipeline.
                  </p>

                  <div className="insight-stat-row">
                    <div>
                      <span>Model loaded</span>
                      <strong>{health?.model_loaded ? 'Yes' : 'No'}</strong>
                    </div>

                    <div>
                      <span>Probability output</span>
                      <strong>
                        {info?.probabilities_supported
                          ? 'Supported'
                          : 'Not supported'}
                      </strong>
                    </div>

                    <div>
                      <span>Classes</span>
                      <strong>{classNames.length || '—'}</strong>
                    </div>
                  </div>
                </article>

                <article className="panel">
                  <div className="panel-header">
                    <div>
                      <h3>Target classes</h3>
                      <p>Raw class labels reported by the model</p>
                    </div>
                  </div>

                  {classNames.length ? (
                    <div className="class-chips">
                      {classNames.map(className => (
                        <span key={className}>
                          {prettyLabel(className)}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <Empty
                      title="Model information unavailable"
                      detail="Connect to the backend to inspect class labels."
                    />
                  )}

                  <div className="panel-footnote">
                    <AlertCircle size={14} />
                    Confirm the target encoding against the training code.
                  </div>
                </article>
              </div>

              <article className="panel features-panel">
                <div className="panel-header">
                  <div>
                    <h3>Expected input features</h3>
                    <p>
                      Names expected by the current form and model pipeline
                    </p>
                  </div>

                  <span className="count-badge">
                    {FEATURES.length} features
                  </span>
                </div>

                <div className="feature-list">
                  {FEATURES.map((feature, index) => (
                    <div key={feature.key}>
                      <span className="feature-index">
                        {String(index + 1).padStart(2, '0')}
                      </span>

                      <span>{feature.name}</span>

                      <span className="feature-type">
                        {feature.type === 'number'
                          ? 'Numeric'
                          : feature.type === 'binary'
                            ? 'Binary'
                            : 'Categorical'}
                      </span>
                    </div>
                  ))}
                </div>
              </article>

              {/* Performance metrics warning removed. */}
            </section>
          )}

          {page === 'Settings' && (
            <section>
              <div className="page-heading">
                <div className="eyebrow">
                  <SettingsIcon size={14} />
                  APPLICATION STATUS
                </div>

                <h1>Settings &amp; connections</h1>

                <p className="page-subtitle">
                  View system connectivity and safe configuration
                  information.
                </p>
              </div>

              <div className="settings-list">
                <article className="panel setting-row">
                  <div className="setting-icon">
                    <Wifi size={19} />
                  </div>

                  <div className="setting-copy">
                    <h3>Backend API</h3>
                    <p>{API_BASE_URL}</p>

                    <small>
                      Configured using VITE_API_BASE_URL, with a local
                      development fallback.
                    </small>
                  </div>

                  <StatusPill
                    ok={
                      health?.status === 'ok' ||
                      !!health?.model_loaded
                    }
                  >
                    {healthLoading
                      ? 'Checking'
                      : health?.model_loaded
                        ? 'Available'
                        : 'Unavailable'}
                  </StatusPill>
                </article>

                <article className="panel setting-row">
                  <div className="setting-icon">
                    <Database size={19} />
                  </div>

                  <div className="setting-copy">
                    <h3>Turso database</h3>
                    <p>Persistent prediction storage</p>

                    <small>
                      Database credentials are kept on the backend
                      and are never displayed here.
                    </small>
                  </div>

                  <StatusPill ok={!!health?.database_connected}>
                    {health?.database_connected
                      ? 'Connected'
                      : 'Unavailable'}
                  </StatusPill>
                </article>

                <article className="panel setting-row">
                  <div className="setting-icon">
                    <ShieldCheck size={19} />
                  </div>

                  <div className="setting-copy">
                    <h3>Saved model</h3>
                    <p>student_outcome_model.pkl</p>

                    <small>
                      Existing SVC pipeline; backend does not retrain
                      the model.
                    </small>
                  </div>

                  <StatusPill ok={!!health?.model_loaded}>
                    {health?.model_loaded ? 'Loaded' : 'Unavailable'}
                  </StatusPill>
                </article>

                <article className="panel setting-row">
                  <div className="setting-icon">
                    <Activity size={19} />
                  </div>

                  <div className="setting-copy">
                    <h3>Application version</h3>
                    <p>1.0.0</p>

                    <small>
                      StudentLens AI · See Potential. Predict Success.
                    </small>
                  </div>

                  <span className="version-tag">v1.0.0</span>
                </article>

                <div className="settings-actions">
                  <button
                    className="btn btn-secondary"
                    onClick={refreshSystem}
                    disabled={healthLoading}
                  >
                    {healthLoading ? (
                      <LoaderCircle className="spin" size={16} />
                    ) : (
                      <RefreshCw size={16} />
                    )}
                    Recheck connections
                  </button>
                </div>
              </div>
            </section>
          )}
        </div>

        <footer className="main-footer">
          <span>© 2026 StudentLens AI</span>
          <span>See Potential. Predict Success.</span>

          <span className="footer-secure">
            <ShieldCheck size={14} />
            Responsible AI
          </span>
        </footer>
      </main>
    </div>
  );
}
