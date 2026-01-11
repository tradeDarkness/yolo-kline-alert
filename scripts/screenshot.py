import asyncio
import os
import time
import sys
from playwright.async_api import async_playwright

# 配置
SAVE_DIR = "data/images"
# 目标网址 (你的 TradingView 图表链接，包含你的指标配置)
TARGET_URL = "https://cn.tradingview.com/chart/499xYnEI/"
# 持久化上下文目录（保存登录状态）
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".playwright-browser-data")
# Chrome 远程调试端口（用于连接已打开的浏览器）
CDP_ENDPOINT = "http://localhost:9222"

def read_input():
    """在异步环境中读取用户输入"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, sys.stdin.readline)

async def run():
    async with async_playwright() as p:
        browser = None
        context = None
        page = None
        
        # 1. 首先尝试连接到已打开的浏览器
        print("🔍 尝试连接到已打开的浏览器...")
        try:
            browser = await p.chromium.connect_over_cdp(CDP_ENDPOINT)
            contexts = browser.contexts
            if len(contexts) > 0:
                context = contexts[0]
                if len(context.pages) > 0:
                    page = context.pages[0]
                else:
                    page = await context.new_page()
                print("✅ 成功连接到已打开的浏览器！")
            else:
                context = await browser.new_context()
                page = await context.new_page()
                print("✅ 连接到浏览器，创建新上下文")
        except Exception as e:
            print(f"⚠️  无法连接到已打开的浏览器: {e}")
            print("💡 提示: 如果你想使用已打开的浏览器，请用以下命令启动 Chrome:")
            print(f"   /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222")
            print("\n   或者让脚本自动打开新浏览器...\n")
            
            # 2. 如果连接失败，使用持久化上下文启动新浏览器
            print("🔐 使用持久化浏览器上下文启动新浏览器...")
            launch_options = {
                "user_data_dir": USER_DATA_DIR,
                "headless": False,
                "viewport": {'width': 1280, 'height': 800}
            }
            
            try:
                # 尝试使用系统 Chrome
                context = await p.chromium.launch_persistent_context(
                    channel="chrome",
                    **launch_options
                )
            except Exception:
                # 如果没有系统 Chrome，使用 Playwright 的 Chromium
                print("⚠️  未找到系统 Chrome，使用 Playwright Chromium")
                context = await p.chromium.launch_persistent_context(
                    **launch_options
                )
            
            # 获取或创建新页面
            if len(context.pages) > 0:
                page = context.pages[0]
            else:
        page = await context.new_page()

        print(f"🚀 正在打开: {TARGET_URL}")
        await page.goto(TARGET_URL)
        
        # 如果连接到已有浏览器，检查当前页面是否是目标页面
        if browser and page.url != TARGET_URL:
            print(f"🚀 正在导航到: {TARGET_URL}")
            await page.goto(TARGET_URL)
        
        # 2. 等待页面加载 (简单粗暴等待，确保K线刷出来)
        print("⏳ 等待页面加载...")
        await page.wait_for_timeout(5000) 
        
        # 检查是否需要登录
        print("💡 提示: 如果是第一次运行，请在浏览器中登录你的 TradingView 账号")
        print("   登录后，下次运行就会自动保持登录状态\n") 

        # 3. 注入 JS 隐藏干扰元素 (根据 TradingView 实际 DOM调整，这里是通用策略)
        # 隐藏左上角信息、右侧工具栏、顶部菜单
        await page.evaluate("""
            const styles = `
                .layout__area--left, .layout__area--top, .layout__area--right { display: none !important; }
                .chart-controls-bar { display: none !important; }
                .floating-toolbar-react-widgets__button { display: none !important; }
            `;
            const styleSheet = document.createElement("style");
            styleSheet.innerText = styles;
            document.head.appendChild(styleSheet);
        """)
        
        print("\n" + "="*60)
        print("📸 半自动截图模式已启动")
        print("="*60)
        print("💡 使用说明:")
        print("   1. 在浏览器中浏览 TradingView 图表")
        print("   2. 当你看到均线密集的形态时")
        print("   3. 回到终端，按 [回车] 键截图")
        print("   4. 输入 'q' 或 'quit' 退出程序")
        print("="*60 + "\n")
        
        saved_count = 0
        
        # 4. 循环等待用户输入并截图
        while True:
            try:
                print("⏳ 等待你的判断... (按回车截图，输入 'q' 退出)")
                user_input = await read_input()
                
                if user_input.strip().lower() in ['q', 'quit', 'exit']:
                    print("👋 退出程序")
                    break
                
                # 用户按了回车，执行截图
            timestamp = int(time.time())
                filename = f"{SAVE_DIR}/{timestamp}_{saved_count}.jpg"
            
                # 截图图表核心区域 (640x640)
            await page.screenshot(path=filename, clip={'x': 320, 'y': 80, 'width': 640, 'height': 640})
            
                saved_count += 1
                print(f"✅ 截图已保存: {filename} (共 {saved_count} 张)\n")
                
            except KeyboardInterrupt:
                print("\n⏹️  用户中断")
                break
            except Exception as e:
                print(f"❌ 处理出错: {e}")

        # 关闭连接（如果是连接的浏览器，只断开连接，不关闭浏览器）
        if browser:
        await browser.close()
        elif context:
            await context.close()
        print(f"\n✅ 采集结束，共保存 {saved_count} 张截图")

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    asyncio.run(run())