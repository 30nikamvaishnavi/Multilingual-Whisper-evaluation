"""Interactive Streamlit dashboard for the Multilingual ASR Evaluation Platform."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.model_manager import is_cuda_available  # noqa: E402
from app.config import (  # noqa: E402
    DEFAULT_RESULTS_CSV_NAME,
    DEFAULT_SUMMARY_JSON_NAME,
    MANIFEST_REQUIRED_COLUMNS,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODELS,
    AppConfig,
)
from app.data.manifest_loader import (  # noqa: E402
    ManifestError,
    append_manifest_record,
    load_manifest,
    save_manifest_records,
    validate_manifest_dataframe,
)
from app.evaluation.evaluator import ASREvaluator  # noqa: E402
from app.evaluation.normalizer import NormalizerConfig  # noqa: E402

st.set_page_config(
    page_title="Multilingual ASR Evaluation Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure crisp Devanagari rendering alongside Latin typography
st.markdown(
    """
    <style>
    html, body, [class*="css"], textarea, input, .stDataFrame {
        font-family: "Inter", "Noto Sans Devanagari", "Kohinoor Devanagari", "Nirmala UI",
            "Mangal", system-ui, -apple-system, sans-serif;
    }
    .transcript-box {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1.02rem;
        line-height: 1.65;
        word-break: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_existing_results(config: AppConfig) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Load persisted evaluation CSV and JSON summary if present and non-empty."""
    csv_path = config.results_dir / DEFAULT_RESULTS_CSV_NAME
    json_path = config.results_dir / DEFAULT_SUMMARY_JSON_NAME

    results_df: pd.DataFrame | None = None
    summary: dict[str, Any] | None = None

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            loaded_df = pd.read_csv(csv_path, encoding="utf-8")
            if not loaded_df.empty:
                results_df = loaded_df
        except Exception:
            results_df = None

    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                loaded_summary = json.load(fh)
            if isinstance(loaded_summary, dict) and loaded_summary.get("total_files", 0) > 0:
                summary = loaded_summary
        except Exception:
            summary = None

    return results_df, summary


def _read_manifest_preview(manifest_path: Path) -> pd.DataFrame:
    """Read raw manifest CSV safely for table preview."""
    if not manifest_path.exists():
        return pd.DataFrame(columns=list(MANIFEST_REQUIRED_COLUMNS))
    try:
        return pd.read_csv(manifest_path, dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=list(MANIFEST_REQUIRED_COLUMNS))


def _sanitize_filename(filename: str) -> str:
    """Keep only a safe filename component when saving uploaded audio."""
    return Path(filename).name.replace(" ", "_")


def _build_model_comparison_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Convert overall summary metrics into a tidy DataFrame for Plotly charts."""
    rows: list[dict[str, Any]] = []
    for model_name, stats in summary.get("overall_metrics", {}).items():
        if stats.get("clips_evaluated", 0) == 0:
            continue
        rows.append(
            {
                "Model": f"Whisper {model_name.capitalize()}",
                "model_key": model_name,
                "Mean WER (%)": stats.get("mean_wer_pct"),
                "Mean CER (%)": stats.get("mean_cer_pct"),
                "Mean Processing Time (s)": stats.get("mean_processing_time_sec"),
                "Mean RTF": stats.get("mean_rtf"),
            }
        )
    return pd.DataFrame(rows)


def _build_language_comparison_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Convert language-wise summary metrics into a tidy DataFrame for grouped charts."""
    rows: list[dict[str, Any]] = []
    for language_name, lang_data in summary.get("language_metrics", {}).items():
        for model_name, stats in lang_data.get("models", {}).items():
            if stats.get("clips_evaluated", 0) == 0:
                continue
            rows.append(
                {
                    "Language": language_name,
                    "Model": f"Whisper {model_name.capitalize()}",
                    "Mean WER (%)": stats.get("mean_wer_pct"),
                    "Mean CER (%)": stats.get("mean_cer_pct"),
                    "Mean Processing Time (s)": stats.get("mean_processing_time_sec"),
                    "Mean RTF": stats.get("mean_rtf"),
                }
            )
    return pd.DataFrame(rows)


def render_overview_section(
    summary: dict[str, Any] | None,
    selected_models: list[str],
    selected_device: str,
) -> None:
    """Section A: Overview metrics and honest evaluation status."""
    st.header("A. Overview")

    total_evaluated = int(summary.get("total_files", 0)) if summary else 0
    if summary is None:
        eval_status = "Not Run"
    elif summary.get("failed_files", 0) == total_evaluated and total_evaluated > 0:
        eval_status = "Failed"
    elif summary.get("partial_files", 0) > 0 or summary.get("failed_files", 0) > 0:
        eval_status = "Completed with Warnings"
    else:
        eval_status = "Completed"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Audio Files Evaluated", total_evaluated)
    col2.metric("Supported Languages", ", ".join(SUPPORTED_LANGUAGES.keys()))
    col3.metric(
        "Available Whisper Models",
        ", ".join(m.capitalize() for m in SUPPORTED_MODELS),
    )
    col4.metric("Selected Device", selected_device.upper())
    col5.metric("Evaluation Status", eval_status)

    if summary is None:
        st.info(
            "No evaluation results are available yet. Add audio clips and reference transcripts "
            "in **Section B**, configure models in **Section C**, and click **Run Evaluation**."
        )
    else:
        norm = summary.get("normalization", {})
        st.caption(
            f"Last evaluation: `{summary.get('generated_at_utc', 'N/A')}` | "
            f"Device used: `{summary.get('device', 'N/A')}` | "
            f"Models: `{', '.join(selected_models or summary.get('models_evaluated', []))}` | "
            f"Normalization enabled: `{norm.get('enabled', False)}` "
            f"(Unicode={norm.get('unicode_form')}, "
            f"Whitespace={norm.get('normalize_whitespace')}, "
            f"RemovePunct={norm.get('remove_punctuation')})"
        )


def render_dataset_section(config: AppConfig) -> None:
    """Section B: Manifest CSV upload, clip addition, validation, and preview."""
    st.header("B. Audio Dataset")

    col_upload, col_add = st.columns(2)

    with col_upload:
        st.subheader("Upload Manifest CSV")
        uploaded_csv = st.file_uploader(
            "Upload a manifest CSV (`audio_path,language,reference_text`)",
            type=["csv"],
            key="manifest_csv_uploader",
        )
        if uploaded_csv is not None:
            try:
                uploaded_df = pd.read_csv(
                    uploaded_csv, dtype=str, keep_default_na=False, encoding="utf-8"
                )
                records, errors = validate_manifest_dataframe(
                    uploaded_df, project_root=config.project_root
                )
                if errors:
                    st.error("Uploaded manifest failed validation:\n- " + "\n- ".join(errors))
                else:
                    if st.button("Apply Uploaded Manifest", key="apply_uploaded_manifest"):
                        save_manifest_records(
                            config.manifest_path, uploaded_df, project_root=config.project_root
                        )
                        st.success(f"Saved validated manifest with {len(records)} record(s).")
                        st.rerun()
                    st.info(f"Validated {len(records)} record(s). Click 'Apply Uploaded Manifest' to save.")
            except Exception as exc:
                st.error(f"Could not parse uploaded CSV: {exc}")

    with col_add:
        st.subheader("Add Audio Clip & Reference Transcript")
        with st.form("add_audio_clip_form", clear_on_submit=True):
            allowed_exts = [ext.lstrip(".") for ext in sorted(SUPPORTED_AUDIO_EXTENSIONS)]
            uploaded_audio = st.file_uploader(
                "Upload audio clip (WAV, MP3, M4A, FLAC)",
                type=allowed_exts,
                key="audio_clip_uploader",
            )
            clip_language = st.selectbox(
                "Spoken Language",
                options=list(SUPPORTED_LANGUAGES.keys()),
                key="clip_language_select",
            )
            reference_transcript = st.text_area(
                "Verified Reference Transcript (use exact Devanagari spelling for Hindi/Marathi)",
                placeholder="उदा. मला मराठी भाषा खूप आवडते / आज मौसम बहुत अच्छा है / Hello world",
                height=100,
                key="clip_reference_input",
            )
            submitted = st.form_submit_button("Add Clip to Manifest")

            if submitted:
                if uploaded_audio is None:
                    st.error("Please select an audio file to upload.")
                elif not reference_transcript.strip():
                    st.error("Reference transcript cannot be empty.")
                else:
                    config.audio_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = _sanitize_filename(uploaded_audio.name)
                    dest_path = config.audio_dir / safe_name
                    dest_path.write_bytes(uploaded_audio.getbuffer())
                    rel_audio_path = f"data/audio/{safe_name}"
                    try:
                        append_manifest_record(
                            manifest_path=config.manifest_path,
                            audio_path=rel_audio_path,
                            language=clip_language,
                            reference_text=reference_transcript,
                            project_root=config.project_root,
                        )
                        st.success(f"Added `{rel_audio_path}` ({clip_language}) to manifest.")
                        st.rerun()
                    except ManifestError as exc:
                        st.error(f"Failed to add record: {exc}")

    st.subheader("Dataset Validation & Manifest Preview")
    manifest_df = _read_manifest_preview(config.manifest_path)

    val_col1, val_col2 = st.columns([1, 3])
    with val_col1:
        if st.button("Validate Manifest", key="validate_manifest_btn"):
            try:
                valid_records = load_manifest(
                    config.manifest_path, project_root=config.project_root, allow_empty=False
                )
                st.success(f"Manifest valid: {len(valid_records)} clip(s) verified.")
            except ManifestError as exc:
                st.error(str(exc))

    with val_col2:
        st.caption(
            f"Manifest path: `{config.manifest_path.relative_to(config.project_root)}` | "
            f"Current rows: **{len(manifest_df)}**"
        )

    if manifest_df.empty:
        st.warning("`data/manifest.csv` currently has no audio records (header only).")
    else:
        st.dataframe(manifest_df, use_container_width=True, hide_index=True)


def render_configuration_section(config: AppConfig) -> tuple[list[str], str]:
    """Section C: Model configuration, normalization controls, and evaluation trigger."""
    st.header("C. Model Configuration & Execution")

    cuda_ready = is_cuda_available()
    device_options = ["auto", "cpu"] + (["cuda"] if cuda_ready else [])

    col1, col2, col3 = st.columns(3)
    with col1:
        use_tiny = st.checkbox("Whisper Tiny (`tiny`)", value=True, key="cfg_use_tiny")
        use_base = st.checkbox("Whisper Base (`base`)", value=True, key="cfg_use_base")
        selected_models = [
            name
            for name, enabled in (("tiny", use_tiny), ("base", use_base))
            if enabled
        ]

    with col2:
        selected_device = st.selectbox(
            "Execution Device",
            options=device_options,
            index=0,
            key="cfg_selected_device",
            help="FP16 is enabled automatically on CUDA GPUs; FP32 is used on CPU.",
        )
        if not cuda_ready:
            st.caption("CUDA GPU not detected in this environment; running in CPU (FP32) mode.")

    with col3:
        language_choice = st.selectbox(
            "Language Filter",
            options=["All", *SUPPORTED_LANGUAGES.keys()],
            index=0,
        )
        language_filter = None if language_choice == "All" else language_choice

    with st.expander("Text Normalization Settings (Documented in Summary Output)", expanded=False):
        norm_col1, norm_col2, norm_col3, norm_col4 = st.columns(4)
        norm_enabled = norm_col1.checkbox("Enable Normalization", value=True)
        norm_ws = norm_col2.checkbox("Normalize Whitespace", value=True, disabled=not norm_enabled)
        norm_punct = norm_col3.checkbox(
            "Remove Punctuation", value=False, disabled=not norm_enabled
        )
        norm_lower = norm_col4.checkbox(
            "Lowercase (Latin only)", value=False, disabled=not norm_enabled
        )

    if st.button("Run Evaluation", type="primary", key="run_evaluation_btn"):
        if not selected_models:
            st.error("Select at least one Whisper model (`tiny` or `base`).")
            return selected_models, selected_device

        normalizer_cfg = NormalizerConfig(
            enabled=norm_enabled,
            unicode_form="NFC",
            normalize_whitespace=norm_ws,
            remove_punctuation=norm_punct,
            lowercase=norm_lower,
        )
        evaluator = ASREvaluator(config=config, normalizer_config=normalizer_cfg)
        progress_bar = st.progress(0.0, text="Preparing evaluation...")

        def _on_progress(done: int, total: int, message: str) -> None:
            ratio = min(max(done / total, 0.0), 1.0) if total > 0 else 1.0
            progress_bar.progress(ratio, text=message)

        try:
            with st.spinner("Running local Whisper inference and calculating WER/CER..."):
                output = evaluator.evaluate(
                    manifest_path=config.manifest_path,
                    models=selected_models,
                    device=selected_device,
                    language_filter=language_filter,
                    output_dir=config.results_dir,
                    on_progress=_on_progress,
                )
            progress_bar.progress(1.0, text="Evaluation completed.")
            if output.summary["successful_files"] == 0 and output.summary["partial_files"] == 0:
                st.error(
                    "Evaluation finished, but all clips failed during transcription. "
                    "Inspect error messages below."
                )
            else:
                st.success(
                    f"Evaluation complete: {output.summary['successful_files']} succeeded, "
                    f"{output.summary['partial_files']} partial, "
                    f"{output.summary['failed_files']} failed."
                )
            st.rerun()
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Evaluation failed: {exc}")

    return selected_models, selected_device


def render_results_section(results_df: pd.DataFrame | None) -> None:
    """Section D: Detailed transcription outputs and Devanagari-safe transcript inspection."""
    st.header("D. Transcription Results")

    if results_df is None or results_df.empty:
        st.info("No transcription results to display. Run an evaluation once audio clips are loaded.")
        return

    display_cols = [
        col
        for col in [
            "audio_filename",
            "language",
            "audio_duration_sec",
            "reference_transcript",
            "tiny_transcription",
            "base_transcription",
            "tiny_wer_pct",
            "base_wer_pct",
            "tiny_cer_pct",
            "base_cer_pct",
            "tiny_processing_time_sec",
            "base_processing_time_sec",
            "tiny_rtf",
            "base_rtf",
            "status",
            "error_message",
        ]
        if col in results_df.columns
    ]
    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

    st.subheader("Transcript Inspector (Unicode / Devanagari View)")
    for idx, row in results_df.iterrows():
        label = f"{row.get('audio_filename', f'Clip {idx + 1}')} — {row.get('language', '')} ({row.get('audio_duration_sec', 'N/A')}s)"
        with st.expander(label, expanded=(idx == 0)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Reference Transcript**")
                st.markdown(
                    f"<div class='transcript-box'>{row.get('reference_transcript', '')}</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                tiny_text = row.get("tiny_transcription")
                tiny_time = row.get("tiny_processing_time_sec")
                st.markdown(
                    f"**Whisper Tiny** ({tiny_time}s)" if pd.notna(tiny_time) else "**Whisper Tiny**"
                )
                st.markdown(
                    f"<div class='transcript-box'>{tiny_text if pd.notna(tiny_text) else 'N/A'}</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                base_text = row.get("base_transcription")
                base_time = row.get("base_processing_time_sec")
                st.markdown(
                    f"**Whisper Base** ({base_time}s)" if pd.notna(base_time) else "**Whisper Base**"
                )
                st.markdown(
                    f"<div class='transcript-box'>{base_text if pd.notna(base_text) else 'N/A'}</div>",
                    unsafe_allow_html=True,
                )


def render_comparison_section(summary: dict[str, Any] | None) -> None:
    """Section E: Visual comparison of Whisper Tiny vs Base WER, CER, latency, and RTF."""
    st.header("E. Model Comparison")

    st.info(
        "**Metric Interpretation Note:** Lower WER and CER indicate better transcription accuracy. "
        "WER and CER measure edit-distance error rates relative to the reference transcript — "
        "they are **not** percentages of correctly recognized words or characters "
        "(insertions can even cause error rates above 100%)."
    )

    if summary is None:
        st.warning("Comparison charts will appear after running an evaluation on your audio dataset.")
        return

    model_df = _build_model_comparison_dataframe(summary)
    lang_df = _build_language_comparison_dataframe(summary)

    if model_df.empty:
        st.warning("No successful model metrics were recorded in the latest evaluation.")
        return

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        fig_wer = px.bar(
            model_df,
            x="Model",
            y="Mean WER (%)",
            color="Model",
            text_auto=".2f",
            title="Overall Mean WER (%) — Tiny vs Base (Lower is Better)",
        )
        fig_wer.update_layout(yaxis_title="Word Error Rate (%)", showlegend=True)
        st.plotly_chart(fig_wer, use_container_width=True)

    with row1_col2:
        fig_cer = px.bar(
            model_df,
            x="Model",
            y="Mean CER (%)",
            color="Model",
            text_auto=".2f",
            title="Overall Mean CER (%) — Tiny vs Base (Lower is Better)",
        )
        fig_cer.update_layout(yaxis_title="Character Error Rate (%)", showlegend=True)
        st.plotly_chart(fig_cer, use_container_width=True)

    if not lang_df.empty:
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            fig_lang_wer = px.bar(
                lang_df,
                x="Language",
                y="Mean WER (%)",
                color="Model",
                barmode="group",
                text_auto=".2f",
                title="Language-wise Mean WER (%) (Lower is Better)",
            )
            fig_lang_wer.update_layout(yaxis_title="Word Error Rate (%)")
            st.plotly_chart(fig_lang_wer, use_container_width=True)

        with row2_col2:
            fig_lang_cer = px.bar(
                lang_df,
                x="Language",
                y="Mean CER (%)",
                color="Model",
                barmode="group",
                text_auto=".2f",
                title="Language-wise Mean CER (%) (Lower is Better)",
            )
            fig_lang_cer.update_layout(yaxis_title="Character Error Rate (%)")
            st.plotly_chart(fig_lang_cer, use_container_width=True)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        fig_time = px.bar(
            model_df,
            x="Model",
            y="Mean Processing Time (s)",
            color="Model",
            text_auto=".3f",
            title="Mean Processing Time per Clip (Seconds)",
        )
        fig_time.update_layout(yaxis_title="Seconds")
        st.plotly_chart(fig_time, use_container_width=True)

    with row3_col2:
        fig_rtf = px.bar(
            model_df,
            x="Model",
            y="Mean RTF",
            color="Model",
            text_auto=".3f",
            title="Mean Real-Time Factor (RTF = Processing Time / Audio Duration)",
        )
        fig_rtf.update_layout(yaxis_title="Real-Time Factor (Lower is Faster)")
        st.plotly_chart(fig_rtf, use_container_width=True)


def render_export_section(config: AppConfig) -> None:
    """Section F: Export detailed CSV and summary JSON reports."""
    st.header("F. Results Export")

    csv_path = config.results_dir / DEFAULT_RESULTS_CSV_NAME
    json_path = config.results_dir / DEFAULT_SUMMARY_JSON_NAME

    has_csv = csv_path.exists() and csv_path.stat().st_size > 0
    has_json = json_path.exists() and json_path.stat().st_size > 0

    if not has_csv and not has_json:
        st.info("No evaluation output files exist yet. Run an evaluation to generate exportable reports.")
        return

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if has_csv:
            st.download_button(
                label="Download Detailed Results (CSV)",
                data=csv_path.read_bytes(),
                file_name=DEFAULT_RESULTS_CSV_NAME,
                mime="text/csv",
                use_container_width=True,
            )
    with exp_col2:
        if has_json:
            st.download_button(
                label="Download Evaluation Summary (JSON)",
                data=json_path.read_bytes(),
                file_name=DEFAULT_SUMMARY_JSON_NAME,
                mime="application/json",
                use_container_width=True,
            )


def main() -> None:
    """Render the Streamlit Multilingual ASR Evaluation Platform."""
    config = AppConfig()

    st.title("Multilingual ASR Evaluation Platform")
    st.markdown(
        "Evaluate and compare multilingual speech recognition models across "
        "**Hindi**, **Marathi**, and **English**."
    )

    with st.sidebar:
        st.subheader("Platform Details")
        st.markdown(
            "- **Models**: Local OpenAI Whisper (`tiny`, `base`)\n"
            "- **Languages**: Hindi (`hi`), Marathi (`mr`), English (`en`)\n"
            "- **Metrics**: WER & CER via `jiwer`, Processing Time, RTF\n"
            "- **Inference**: 100% local & open-source (no paid APIs)"
        )
        st.divider()
        st.caption(f"Project root: `{config.project_root.name}`")

    results_df, summary = _load_existing_results(config)

    default_device = "cuda" if is_cuda_available() else "cpu"
    active_models = [
        name
        for name, key in (("tiny", "cfg_use_tiny"), ("base", "cfg_use_base"))
        if st.session_state.get(key, True)
    ] or list(SUPPORTED_MODELS)
    active_device = str(st.session_state.get("cfg_selected_device", default_device))
    if active_device == "auto":
        active_device = default_device

    render_overview_section(summary, active_models, active_device)
    st.divider()
    render_dataset_section(config)
    st.divider()
    render_configuration_section(config)
    st.divider()
    render_results_section(results_df)
    st.divider()
    render_comparison_section(summary)
    st.divider()
    render_export_section(config)


if __name__ == "__main__":
    main()
