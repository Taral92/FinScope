from finscope.text import html_to_text

# A miniature inline-XBRL filing: hidden XBRL header, script, style, prose, and a table.
FAKE_FILING = """\
<html>
<head>
  <style>p { font-family: Times; }</style>
  <script>console.log("tracking");</script>
</head>
<body>
  <div style="display: none">
    <ix:header><ix:hidden>dei:DocumentFiscalPeriodFocus Q3</ix:hidden></ix:header>
  </div>
  <p>Tesla   designs,
     develops&nbsp;and   <span>sells</span> electric vehicles.</p>
  <p></p><p></p><p></p>
  <p>Revenue grew this quarter.</p>
  <table>
    <tr><th>Item</th><th></th><th>Amount</th></tr>
    <tr><td>Revenue</td><td>&nbsp;</td><td>28,095</td></tr>
  </table>
</body>
</html>
"""


def test_hidden_xbrl_script_and_style_are_removed() -> None:
    text = html_to_text(FAKE_FILING)

    assert "DocumentFiscalPeriodFocus" not in text
    assert "tracking" not in text
    assert "font-family" not in text


def test_paragraph_text_is_kept_with_whitespace_cleaned() -> None:
    lines = html_to_text(FAKE_FILING).splitlines()

    # Source line breaks, repeated spaces and &nbsp; all collapse to single spaces.
    assert "Tesla designs, develops and sells electric vehicles." in lines
    assert "Revenue grew this quarter." in lines


def test_table_rows_become_lines_with_cells_joined() -> None:
    lines = html_to_text(FAKE_FILING).splitlines()

    assert "Item | Amount" in lines  # empty padding cells are dropped
    assert "Revenue | 28,095" in lines


def test_whitespace_is_normalized() -> None:
    text = html_to_text(FAKE_FILING)

    assert "\xa0" not in text
    assert "  " not in text
    assert "\n\n\n" not in text  # never more than one blank line in a row
    assert text == text.strip()
