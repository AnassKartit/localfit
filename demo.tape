# localfit demo
Output demo.gif
Set FontSize 15
Set Width 1100
Set Height 650
Set Theme "Catppuccin Mocha"
Set Padding 25
Set TypingSpeed 40ms

# Boot screen
Type@50ms "python3 /tmp/localfit_demo.py"
Enter
Sleep 4s
Screenshot screenshots/boot.png

Enter
Sleep 1s

# Simulate
Type@50ms "python3 /tmp/localfit_demo.py --simulate '35b q3'"
Enter
Sleep 3s
Screenshot screenshots/simulate.png
Sleep 1s

# Fetch from HuggingFace
Type@50ms "python3 /tmp/localfit_demo.py --fetch unsloth/Qwen3.5-35B-A3B-GGUF"
Enter
Sleep 5s
Screenshot screenshots/fetch.png
Type "q"
Enter
Sleep 500ms

# Health
Type@50ms "python3 /tmp/localfit_demo.py --health"
Enter
Sleep 3s
Screenshot screenshots/health.png
Type "q"
Enter
Sleep 500ms

# Specs
Type@50ms "python3 /tmp/localfit_demo.py --specs"
Enter
Sleep 2s
Screenshot screenshots/specs.png

# Config
Type@50ms "python3 /tmp/localfit_demo.py --config opencode"
Enter
Sleep 2s
Screenshot screenshots/config.png
Sleep 1s
