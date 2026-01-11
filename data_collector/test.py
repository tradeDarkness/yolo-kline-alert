import subprocess

def test_switch():
    symbol = "BTCUSDT.P"
    applescript = f'''
    tell application "TradingView" to activate
    delay 1
    tell application "System Events"
        keystroke "{symbol}"
        delay 0.5
        key code 36
    end tell
    '''
    subprocess.run(["osascript", "-e", applescript])

test_switch()