"""
Nervous Machine One-Pager PDF Generator
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
import os

OUTPUT_PATH = "/mnt/user-data/outputs/nervous_machine_onepager.pdf"

# ─── Colors ───
DARK_BG = HexColor("#0A0E17")
CARD_BG = HexColor("#111827")
ACCENT = HexColor("#10B981")       # Emerald green
ACCENT_DIM = HexColor("#065F46")
TEXT_PRIMARY = HexColor("#F9FAFB")
TEXT_SECONDARY = HexColor("#9CA3AF")
TEXT_MUTED = HexColor("#6B7280")
BORDER = HexColor("#1F2937")
HIGHLIGHT = HexColor("#059669")
WARM = HexColor("#F59E0B")         # Amber for warnings/contrast
CYAN = HexColor("#06B6D4")

W, H = letter  # 612 x 792


def draw_rounded_rect(c, x, y, w, h, r, fill_color=None, stroke_color=None, stroke_width=0.5):
    """Draw a rounded rectangle."""
    p = c.beginPath()
    p.roundRect(x, y, w, h, r)
    p.close()
    if fill_color:
        c.setFillColor(fill_color)
    if stroke_color:
        c.setStrokeColor(stroke_color)
        c.setLineWidth(stroke_width)
    if fill_color and stroke_color:
        c.drawPath(p, fill=1, stroke=1)
    elif fill_color:
        c.drawPath(p, fill=1, stroke=0)
    elif stroke_color:
        c.drawPath(p, fill=0, stroke=1)


def draw_icon_circle(c, x, y, r, color):
    """Draw a small colored circle as icon placeholder."""
    c.setFillColor(color)
    c.circle(x, y, r, fill=1, stroke=0)


def create_onepager():
    c = canvas.Canvas(OUTPUT_PATH, pagesize=letter)

    # ─── Full page dark background ───
    c.setFillColor(DARK_BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # ─── Top accent line ───
    c.setFillColor(ACCENT)
    c.rect(0, H - 4, W, 4, fill=1, stroke=0)

    margin = 40
    content_w = W - 2 * margin

    # ─── Header ───
    y = H - 50

    # Logo / Brand
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin, y, "NERVOUS MACHINE")

    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica", 9)
    c.drawRightString(W - margin, y + 10, "nervousmachine.com")
    c.drawRightString(W - margin, y - 2, "context.nervousmachine.com")

    # ─── Positioning Statement ───
    y -= 32
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 10.5)
    c.drawString(margin, y, "For fluency, leverage an LLM.")
    c.drawString(margin + 185, y, "For exploration, leverage a world model.")

    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(margin + 445, y, "For certainty,")

    y -= 16
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(margin, y, "leverage Nervous Machine.")

    # ─── Tagline ───
    y -= 24
    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "The World Is Full of Edge Cases.")

    y -= 16
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 9.5)
    c.drawString(
        margin, y, "A causal learning engine for devices that need to learn from the real world — not just predict from the average.")

    # ─── Thin separator ───
    y -= 14
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(margin, y, W - margin, y)

    # ═══════════════════════════════════════════════════
    # SECTION 1: THE PROBLEM
    # ═══════════════════════════════════════════════════
    y -= 22
    c.setFillColor(WARM)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(margin, y, "THE VARIATION PROBLEM")

    y -= 14
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 8)

    lines = [
        "Foundation models optimize for the mean. This makes them fluent — they sound right almost all the time. But fluency and",
        "operational reliability are different things. Every deployed device operates in a unique environment. A robot encounters",
        "conditions the training data never covered. A battery in Houston degrades differently than one in Munich. A satellite's drag",
        "deviates from the model because of solar activity nobody predicted. These aren't bugs — they're the normal operating condition.",
        "",
        "The ceiling on model capability is rising fast. The floor — the minimum reliability in a specific context — is barely moving.",
        "That gap is where deployed systems fail confidently. World models have a similar problem: exploration is expensive, risky,",
        "and the sim-to-real gap persists because simulation compresses the same variance that makes reality unpredictable.",
    ]
    for line in lines:
        c.drawString(margin, y, line)
        y -= 11

    # ═══════════════════════════════════════════════════
    # SECTION 2: THE ENGINE — Two column layout
    # ═══════════════════════════════════════════════════
    y -= 8
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(margin, y, "THE CAUSAL LEARNING ENGINE")

    y -= 8
    col_w = (content_w - 20) / 2
    col1_x = margin
    col2_x = margin + col_w + 20

    # ─── Left column: Local Loop ───
    box_h = 132
    y_box = y - box_h
    draw_rounded_rect(c, col1_x, y_box, col_w, box_h, 6,
                      fill_color=CARD_BG, stroke_color=BORDER)

    inner_y = y - 16
    c.setFillColor(CYAN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col1_x + 14, inner_y, "LOCAL LOOP — On-Device Learning")

    inner_y -= 16
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 8)
    local_lines = [
        "Each device runs a lightweight causal engine",
        "that learns from its own error signals:",
        "",
        "  Predict > Observe reality > Measure error",
        "  > Update certainty > Repeat",
        "",
        "No retraining. No cloud round-trip. The device",
        "builds a local world model that tracks what it",
        "knows, how confidently, and where the gaps are.",
        "When certainty drops, curiosity triggers flag",
        "what it doesn't know.",
    ]
    for line in local_lines:
        c.drawString(col1_x + 14, inner_y, line)
        inner_y -= 10

    # ─── Right column: Outer Loop ───
    draw_rounded_rect(c, col2_x, y_box, col_w, box_h, 6,
                      fill_color=CARD_BG, stroke_color=BORDER)

    inner_y = y - 16
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col2_x + 14, inner_y, "OUTER LOOP — Fleet Learning")

    inner_y -= 16
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 8)
    outer_lines = [
        "Devices share learned causal vectors (~1KB)",
        "across the fleet — not raw data.",
        "",
        "  \"Thermal stress above X at this RPM causes",
        "   bearing failure\" propagates. Your telemetry",
        "   stays local.",
        "",
        "When one device confirms a finding, certainty",
        "rises across the network. When another device",
        "contradicts it, the system holds the boundary",
        "where a 'universal' finding is context-specific.",
    ]
    for line in outer_lines:
        c.drawString(col2_x + 14, inner_y, line)
        inner_y -= 10

    y = y_box - 8

    # ─── Learning rate equation callout ───
    eq_h = 28
    draw_rounded_rect(c, margin, y - eq_h, content_w,
                      eq_h, 4, fill_color=ACCENT_DIM)

    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(margin + 14, y - 12, "ADAPTIVE LEARNING RATE")
    c.setFont("Helvetica", 8.5)
    # Center equation over the middle (Sample Efficient) card: margin + card_w + 10 + card_w/2 ≈ 306
    c.drawCentredString(306, y - 12, "n(Z) = 1 / (1 + e^(10(Z - 0.5)))")
    c.setFillColor(HexColor("#A7F3D0"))
    c.setFont("Helvetica", 7.5)
    c.drawString(margin + 330, y - 12,
                 "Low certainty = learn fast.  High certainty = resist noise.")
    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica", 7)
    c.drawString(margin + 14, y - 24,
                 "At Z=0.1: n=0.99 (aggressive updates)     At Z=0.5: n=0.50 (balanced)     At Z=0.9: n=0.01 (stable, noise-resistant)")

    y = y - eq_h - 10

    # ═══════════════════════════════════════════════════
    # SECTION 3: LEAN INTELLIGENCE
    # ═══════════════════════════════════════════════════
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(margin, y, "LEAN INTELLIGENCE")

    y -= 6

    # Three mini-cards in a row
    card_w = (content_w - 20) / 3
    card_h = 70
    y_cards = y - card_h

    cards = [
        ("ENERGY EFFICIENT", CYAN, [
            "No retraining required.",
            "Causal vectors are ~1KB.",
            "Runs on MCU, Raspberry Pi,",
            "or edge GPU. No cloud",
            "dependency for learning.",
        ]),
        ("SAMPLE EFFICIENT", ACCENT, [
            "Learns from single error",
            "signals, not millions of",
            "samples. One contradiction",
            "can shift certainty. One",
            "confirmation strengthens it.",
        ]),
        ("PRIVACY PRESERVING", WARM, [
            "Raw data never leaves the",
            "device. Only learned causal",
            "relationships propagate.",
            "IP stays local. Lessons",
            "travel across the fleet.",
        ]),
    ]

    for i, (title, color, lines_list) in enumerate(cards):
        cx = margin + i * (card_w + 10)
        draw_rounded_rect(c, cx, y_cards, card_w, card_h, 5,
                          fill_color=CARD_BG, stroke_color=BORDER)

        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(cx + 10, y - 14, title)

        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 7.5)
        ly = y - 28
        for line in lines_list:
            c.drawString(cx + 10, ly, line)
            ly -= 10

    y = y_cards - 10

    # ═══════════════════════════════════════════════════
    # SECTION 4: DEPLOYMENT
    # ═══════════════════════════════════════════════════
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(margin, y, "SEAMLESS DEPLOYMENT")

    y -= 14
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 8)

    deploy_lines = [
        "The Nervous Machine protocol deploys as a lightweight runtime alongside your existing models and firmware. No changes to your",
        "model architecture. No new training pipeline. Drop in the SKILL.md, connect via MCP, and your device starts learning from reality.",
    ]
    for line in deploy_lines:
        c.drawString(margin, y, line)
        y -= 11

    y -= 4

    # Hardware compatibility bar
    hw_h = 24
    draw_rounded_rect(c, margin, y - hw_h, content_w, hw_h, 4,
                      fill_color=CARD_BG, stroke_color=BORDER)

    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(margin + 12, y - 10, "RUNS ON:")

    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Helvetica", 7.5)
    hw_items = ["MCU / Cortex-M", "Raspberry Pi",
                "Jetson / Edge GPU", "Cloud VM", "Any MCP Client"]
    x_pos = margin + 70
    for item in hw_items:
        # Dot
        c.setFillColor(ACCENT)
        c.circle(x_pos, y - 7, 2, fill=1, stroke=0)
        c.setFillColor(TEXT_PRIMARY)
        c.drawString(x_pos + 6, y - 10, item)
        x_pos += 95

    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica", 7)
    c.drawString(margin + 12, y - 21, "PROTOCOL:")
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 7)
    c.drawString(margin + 70, y - 21,
                 "39 MCP tools  |  SKILL.md drop-in  |  Works with Claude, Gemini, Cursor, any MCP-compatible agent or runtime")

    y = y - hw_h - 10

    # ═══════════════════════════════════════════════════
    # SECTION 5: USE CASES
    # ═══════════════════════════════════════════════════
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(margin, y, "WHERE IT MATTERS")

    y -= 6
    uc_h = 44
    uc_w = (content_w - 15) / 4
    y_uc = y - uc_h

    use_cases = [
        ("Robotics & Mfg", [
            "Robots learn local physics.",
            "Fleet shares what works.",
            "Stops confident failures.",
        ]),
        ("Battery & Energy", [
            "Degradation varies by unit.",
            "Learn per-cell behavior.",
            "Optimize charge cycles.",
        ]),
        ("Autonomous Systems", [
            "Sim-to-real gap narrows",
            "with every deployment.",
            "Edge cases become signal.",
        ]),
        ("IoT & Sensors", [
            "Distributed signals, one",
            "certainty graph. Anomaly",
            "detection that learns.",
        ]),
    ]

    for i, (title, lines_list) in enumerate(use_cases):
        ux = margin + i * (uc_w + 5)
        draw_rounded_rect(c, ux, y_uc, uc_w, uc_h, 4,
                          fill_color=CARD_BG, stroke_color=BORDER)

        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(ux + 8, y - 13, title)

        c.setFillColor(TEXT_MUTED)
        c.setFont("Helvetica", 7)
        ly = y - 26
        for line in lines_list:
            c.drawString(ux + 8, ly, line)
            ly -= 9.5

    y = y_uc - 12

    # ═══════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(margin, y, W - margin, y)

    y -= 14
    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(margin, y, "Deploy the nervous system for your devices.")

    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(margin, y - 12,
                 "Seamless. Energy efficient. Learns from every edge case.")

    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(W - margin, y, "context.nervousmachine.com")
    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawRightString(
        W - margin, y - 12, "heidi@nervousmachine.com  |  CLI Demo: github.com/Nervous-Machine/causalAI-demo")

    # ─── Bottom accent line ───
    c.setFillColor(ACCENT)
    c.rect(0, 0, W, 3, fill=1, stroke=0)

    c.save()
    print(f"PDF created: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_onepager()
