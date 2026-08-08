from enum import Enum

css = """
.positive-pnl {
    color: green !important;
    font-weight: bold;
}
.positive-bg {
    background-color: green !important;
    font-weight: bold;
}
.negative-bg {
    background-color: red !important;
    font-weight: bold;
}
.negative-pnl {
    color: red !important;
    font-weight: bold;
}
.dataframe-fix-small .table-wrap {
min-height: 150px;
max-height: 150px;
}
.dataframe-fix .table-wrap {
min-height: 200px;
max-height: 200px;
}
/* ID-based selectors (highest specificity) for transactions tables */
#transactions-warren table tbody tr,
#transactions-george table tbody tr,
#transactions-ray table tbody tr,
#transactions-cathie table tbody tr {
    height: 11px !important;
    line-height: 11px !important;
}
#transactions-warren table td,
#transactions-george table td,
#transactions-ray table td,
#transactions-cathie table td {
    padding: 0px 2px !important;
    font-size: 9px !important;
    height: 11px !important;
    line-height: 11px !important;
}
#transactions-warren table th,
#transactions-george table th,
#transactions-ray table th,
#transactions-cathie table th {
    padding: 0px 2px !important;
    font-size: 9px !important;
    height: 11px !important;
}
/* Options positions table - compact all columns, small font */
#options-cathie table tbody tr,
#options-cathie table thead tr {
    height: 11px !important;
    line-height: 11px !important;
}
#options-cathie table td,
#options-cathie table th {
    font-size: 9px !important;
    padding: 0px 2px !important;
    height: 11px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 52px !important;
}
/* Expiry col (5) - same width as Opened/Closed date cols */
#options-cathie table th:nth-child(5),
#options-cathie table td:nth-child(5) {
    max-width: 52px !important;
    min-width: 52px !important;
}
/* Closed col (8) - same width as Opened */
#options-cathie table th:nth-child(8),
#options-cathie table td:nth-child(8) {
    max-width: 52px !important;
    min-width: 52px !important;
}
/* Holdings table - stock traders */
.dataframe-fix-small table tbody tr,
.dataframe-fix-small table tbody tr td,
.dataframe-fix-small table thead tr th {
    height: 11px !important;
    line-height: 11px !important;
    font-size: 9px !important;
    padding: 0px 2px !important;
}
footer{display:none !important}
"""


js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

class Color(Enum):
    RED = "#dd0000"
    GREEN = "#00dd00"
    YELLOW = "#dddd00"
    BLUE = "#0000ee"
    MAGENTA = "#aa00dd"
    CYAN = "#00dddd"
    WHITE = "#87CEEB"