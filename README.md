# Hyperliquid Monitor Plus

<div align="center">

[![PyPI version](https://badge.fury.io/py/hyperliquid-monitor-plus.svg)](https://badge.fury.io/py/hyperliquid-monitor-plus)
[![Python](https://img.shields.io/pypi/pyversions/hyperliquid-monitor-plus.svg)](https://pypi.org/project/hyperliquid-monitor-plus/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/hyperliquid-monitor-plus)](https://pepy.tech/project/hyperliquid-monitor-plus)

> **💝 Love this project? [Support the developer](https://github.com/Pezhman5252/hyperliquid_monitor_plus#-support-the-developer) to keep it growing!**

**Advanced Production-Ready Whale Tracking & Intelligence System for Hyperliquid DEX**

A comprehensive, enterprise-grade Python library for real-time monitoring, tracking, and analyzing whale wallet transactions on Hyperliquid DEX. Built with production-ready features including ML-powered intelligence, cross-chain tracking, advanced analytics, PnL management, and real-time dashboards.

---

## 🚀 Key Features

### Core Monitoring
- **Real-time Trade Tracking**: Monitor whale wallet fills (executed trades) with WebSocket streaming
- **Multi-Address Support**: Track multiple wallet addresses simultaneously
- **Thread-Safe Operations**: Production-ready with concurrent access handling
- **Automatic Database Storage**: SQLite-based persistent storage with optimized queries
- **Correct Side Mapping**: Properly handles Hyperliquid's side encoding (A=SELL, B=BUY)

### Advanced Analytics & Intelligence
- **Whale Tier Classification**: Automatic categorization (Minnow, Dolphin, Whale, Mega Whale, Giant Whale)
- **Market Impact Analysis**: Predict price impact, slippage, and optimal entry timing
- **Volume Anomaly Detection**: Identify unusual trading volumes and patterns
- **Correlation Engine**: Track correlations between whale movements and market behavior
- **Sentiment Analysis**: Integrate social sentiment into trading decisions
- **Pattern Recognition**: ML-powered detection of recurring whale behaviors

### PnL & Position Management
- **Real-Time PnL Tracking**: Automatic profit/loss calculation per coin and portfolio
- **Position Tracking**: Live position monitoring with liquidation risk alerts
- **Unrealized PnL**: Dynamic calculation with current market prices
- **Portfolio Analytics**: Comprehensive statistics, win rates, best/worst performers
- **Risk Assessment**: Leverage tracking, margin monitoring, liquidation distance

### Smart Alert System
- **Flexible Conditions**: Custom alerts based on volume, price, coins, addresses
- **Alert Levels**: Info, Warning, Critical with different notification strategies
- **Filter System**: Advanced trade filtering by multiple criteria
- **Alert History**: Track and analyze past alerts
- **Callback Support**: Integrate with Discord, Telegram, or custom notifications

### ML & Advanced Features (Phase 3)
- **Machine Learning Models**: Predict whale behavior and price movements
- **Cross-Chain Analytics**: Track whale activity across multiple blockchains
- **Portfolio Optimization**: AI-powered position sizing and risk management
- **Real-Time Dashboard**: Beautiful web interface with live updates
- **Predictive Intelligence**: Forecast whale movements and market trends

### Notifications & Integrations
- **Discord Integration**: Automated alerts to Discord channels
- **Telegram Bots**: Real-time notifications via Telegram
- **Custom Webhooks**: Integrate with any external service
- **Report Generation**: Automated HTML, PDF, and JSON reports

---

## 📦 Installation

### Basic Installation

```bash
pip install hyperliquid-monitor-plus
```

### Installation with Optional Features

```bash
# For visualization support (charts, graphs)
pip install hyperliquid-monitor-plus[viz]

# For machine learning features
pip install hyperliquid-monitor-plus[ml]

# For web dashboard
pip install hyperliquid-monitor-plus[web]

# For Discord/Telegram notifications
pip install hyperliquid-monitor-plus[notifications]

# For development (testing, linting, type checking)
pip install hyperliquid-monitor-plus[dev]

# Install everything
pip install hyperliquid-monitor-plus[all]
```

### From Source

```bash
git clone https://github.com/Pezhman5252/hyperliquid_monitor_plus.git
cd hyperliquid_monitor_plus
pip install -e .
```

---

## 🎯 Quick Start

### 1. Basic Whale Monitoring

```python
from hyperliquid_monitor_plus import HyperliquidMonitor, Trade

# Define callback for trades
def on_trade(trade: Trade):
    print(f"🐋 Whale Trade Detected!")
    print(f"   Coin: {trade.coin}")
    print(f"   Side: {trade.side}")
    print(f"   Size: {trade.size} @ ${trade.price}")
    print(f"   Value: ${trade.size * trade.price:,.2f}")
    print(f"   Address: {trade.address[:10]}...")
    if trade.leverage:
        print(f"   Leverage: {trade.leverage}x")
    print()

# Initialize monitor
monitor = HyperliquidMonitor(
    addresses=[
        "0x1234567890123456789012345678901234567890",  # Replace with actual whale address
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"   # Add more addresses
    ],
    db_path="whale_trades.db",  # Optional: store trades in database
    callback=on_trade,
    track_pnl=True  # Enable automatic PnL tracking
)

# Start monitoring
monitor.start()
```

### 2. Monitor with Alert System

```python
from hyperliquid_monitor_plus import (
    HyperliquidMonitor,
    AlertManager,
    AlertCondition,
    AlertLevel,
    AlertType
)

# Create alert callback
def on_alert(alert):
    print(f"🚨 ALERT [{alert.level}]: {alert.message}")
    if alert.level == "CRITICAL":
        # Send urgent notification
        print(f"   URGENT: {alert.trade.coin} - ${alert.trade.size * alert.trade.price:,.0f}")

# Setup alert manager
alert_manager = AlertManager(alert_callback=on_alert)

# Add alert conditions
alert_manager.add_condition(
    AlertCondition(
        name="Large BTC Trade",
        alert_type=AlertType.LARGE_TRADE,
        level=AlertLevel.CRITICAL,
        min_volume_usd=1_000_000,  # $1M+ trades
        coins=["BTC"]
    )
)

alert_manager.add_condition(
    AlertCondition(
        name="ETH Whale Activity",
        alert_type=AlertType.WHALE_ACTIVITY,
        level=AlertLevel.WARNING,
        min_volume_usd=500_000,  # $500K+ trades
        coins=["ETH"]
    )
)

alert_manager.add_condition(
    AlertCondition(
        name="Any SELL Over $250K",
        alert_type=AlertType.VOLUME_SPIKE,
        level=AlertLevel.INFO,
        min_volume_usd=250_000,
        sides=["SELL"]
    )
)

# Start monitoring with alerts
monitor = HyperliquidMonitor(
    addresses=["0x1234..."],
    db_path="whale_trades.db",
    alert_callback=on_alert,
    track_pnl=True
)

monitor.start()
```

### 3. Advanced PnL Tracking

```python
from hyperliquid_monitor_plus import HyperliquidMonitor, PnLManager

# Create PnL callback
def on_pnl_update(pnl_record):
    if pnl_record.pnl > 0:
        print(f"✅ Profitable Close: {pnl_record.coin}")
    else:
        print(f"❌ Loss Taken: {pnl_record.coin}")
    print(f"   PnL: ${pnl_record.pnl:,.2f}")
    print(f"   Entry: ${pnl_record.entry_price:.2f} -> Exit: ${pnl_record.exit_price:.2f}")
    print(f"   Size: {pnl_record.size}")

# Initialize with PnL tracking
monitor = HyperliquidMonitor(
    addresses=["0x1234..."],
    db_path="whale_trades.db",
    pnl_callback=on_pnl_update,
    track_pnl=True
)

# Access PnL manager
pnl_manager = monitor.pnl_manager

# Start monitoring
monitor.start()

# In another thread/after some trades, you can query:
# Get portfolio statistics
stats = pnl_manager.get_statistics()
print(f"Total Realized PnL: ${stats['total_realized_pnl']:,.2f}")
print(f"Total Unrealized PnL: ${stats['total_unrealized_pnl']:,.2f}")
print(f"Net PnL (after fees): ${stats['net_pnl']:,.2f}")
print(f"Win Rate: {stats['win_rate']:.1f}%")

# Get specific coin PnL
btc_pnl = pnl_manager.get_coin_pnl("BTC")
if btc_pnl:
    print(f"BTC Trades: {btc_pnl.trades_count}")
    print(f"BTC Realized PnL: ${btc_pnl.realized_pnl:,.2f}")
    print(f"BTC Win/Loss: {btc_pnl.winning_trades}/{btc_pnl.losing_trades}")

# Generate summary report
report = pnl_manager.get_summary_report()
print(report)
```

### 4. Live Position Tracking

```python
from hyperliquid_monitor_plus import PositionTracker

# Create position change callback
def on_position_change(change):
    print(f"📍 Position {change.event_type}: {change.coin}")
    if change.old_position:
        print(f"   Old: {change.old_position.side} {change.old_position.size}")
    if change.new_position:
        print(f"   New: {change.new_position.side} {change.new_position.size}")
    print(f"   Size Change: {change.size_change:+.4f}")
    print(f"   PnL Change: ${change.pnl_change:+,.2f}")

# Initialize position tracker
tracker = PositionTracker(
    address="0x1234567890123456789012345678901234567890",
    position_callback=on_position_change
)

# Fetch current positions
snapshot = tracker.fetch_positions()
print(f"Total Unrealized PnL: ${snapshot.total_unrealized_pnl:,.2f}")
print(f"Margin Used: ${snapshot.total_margin_used:,.2f}")
print(f"Account Value: ${snapshot.account_value:,.2f}")

# Get open positions
open_positions = tracker.get_open_positions()
for coin, position in open_positions.items():
    print(f"{coin}: {position.side} {position.size} @ ${position.entry_price:.2f}")
    print(f"  Mark Price: ${position.mark_price:.2f}")
    print(f"  Unrealized PnL: ${position.unrealized_pnl:,.2f} ({position.pnl_percentage:+.2f}%)")
    print(f"  Leverage: {position.leverage}x")
    if position.liquidation_price:
        print(f"  Liquidation: ${position.liquidation_price:.2f} ({position.distance_to_liquidation:.1f}% away)")

# Check risky positions
risky = tracker.get_risky_positions(threshold=15.0)  # 15% from liquidation
if risky:
    print(f"⚠️ WARNING: {len(risky)} position(s) at risk!")
    for pos in risky:
        print(f"  {pos.coin}: {pos.distance_to_liquidation:.1f}% to liquidation")

# Get comprehensive statistics
stats = tracker.get_statistics()
print(f"Open Positions: {stats['open_positions_count']}")
print(f"Best Position: {stats['best_position']} (${stats['best_pnl']:,.2f})")
print(f"Worst Position: {stats['worst_position']} (${stats['worst_pnl']:,.2f})")
```

---

## 📚 Comprehensive Usage Guide

### Working with the Database

```python
from hyperliquid_monitor_plus import Database, init_database

# Initialize database
db_path = "whale_trades.db"
db = init_database(db_path)

# Or use Database class directly
db = Database(db_path)

# Query recent trades
recent_trades = db.get_recent_trades(limit=10)
for trade in recent_trades:
    print(f"{trade.timestamp} - {trade.coin} {trade.side} {trade.size} @ ${trade.price}")

# Query trades by coin
btc_trades = db.get_trades_by_coin("BTC", limit=50)
print(f"Found {len(btc_trades)} BTC trades")

# Query trades by address
whale_trades = db.get_trades_by_address("0x1234...", limit=100)

# Query trades by date range
from datetime import datetime, timedelta
start_date = datetime.now() - timedelta(days=7)
end_date = datetime.now()
weekly_trades = db.get_trades_by_date_range(start_date, end_date)

# Get trade statistics
stats = db.get_trade_statistics(coin="ETH")
print(f"Total ETH Volume: ${stats['total_volume']:,.2f}")
print(f"Average Trade Size: {stats['avg_size']:.4f}")
print(f"Largest Trade: {stats['max_size']:.4f} @ ${stats['max_price']:.2f}")

# Get unique coins and addresses
coins = db.get_unique_coins()
addresses = db.get_unique_addresses()

# Close database when done
db.close()
```

### Advanced Alert Configuration

```python
from hyperliquid_monitor_plus import (
    AlertManager,
    AlertCondition,
    AlertLevel,
    AlertType,
    TradeFilter
)

alert_manager = AlertManager()

# Multi-coin large trade alert
alert_manager.add_condition(
    AlertCondition(
        name="Major Altcoin Movement",
        alert_type=AlertType.LARGE_TRADE,
        level=AlertLevel.WARNING,
        min_volume_usd=100_000,
        coins=["SOL", "AVAX", "ARB", "OP"],
        message_template="🔔 Large {side} on {coin}: {size} @ ${price} (${volume_usd:,.0f})"
    )
)

# Specific address monitoring
alert_manager.add_condition(
    AlertCondition(
        name="Known Whale Activity",
        alert_type=AlertType.WHALE_ACTIVITY,
        level=AlertLevel.CRITICAL,
        addresses=["0xspecificwhale..."],
        min_volume_usd=50_000,
        message_template="🐋 Known whale {address} {side} {coin}: ${volume_usd:,.0f}"
    )
)

# Liquidation monitoring (based on side patterns)
alert_manager.add_condition(
    AlertCondition(
        name="Possible Liquidations",
        alert_type=AlertType.LIQUIDATION,
        level=AlertLevel.INFO,
        min_volume_usd=200_000,
        sides=["SELL"],
        message_template="⚡ Potential liquidation: {coin} - ${volume_usd:,.0f}"
    )
)

# Enable/disable conditions dynamically
alert_manager.disable_condition("Known Whale Activity")
alert_manager.enable_condition("Known Whale Activity")

# Update condition parameters
alert_manager.update_condition(
    "Major Altcoin Movement",
    min_volume_usd=150_000,
    level=AlertLevel.CRITICAL
)

# Get alert statistics
stats = alert_manager.get_statistics()
print(f"Total Conditions: {stats['total_conditions']}")
print(f"Enabled: {stats['enabled_conditions']}")
print(f"Total Triggered: {stats['total_triggered']}")

# Get alert history
recent_alerts = alert_manager.get_history(limit=20)
for alert in recent_alerts:
    print(f"[{alert.timestamp}] {alert.level}: {alert.message}")
```

### Strategy Analysis

```python
from hyperliquid_monitor_plus import StrategyAnalyzer

# Initialize analyzer with trade history
analyzer = StrategyAnalyzer(address="0x1234...")

# Analyze trading patterns
metrics = analyzer.analyze_trading_strategy(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

print("📊 Strategy Metrics:")
print(f"  Total Trades: {metrics.total_trades}")
print(f"  Win Rate: {metrics.win_rate:.1f}%")
print(f"  Profit Factor: {metrics.profit_factor:.2f}")
print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"  Max Drawdown: {metrics.max_drawdown:.1f}%")
print(f"  Average Trade Duration: {metrics.avg_trade_duration}")

# Identify trade patterns
patterns = analyzer.identify_patterns()
for pattern in patterns:
    print(f"Pattern: {pattern.type}")
    print(f"  Frequency: {pattern.frequency}")
    print(f"  Success Rate: {pattern.success_rate:.1f}%")
    print(f"  Avg PnL: ${pattern.avg_pnl:,.2f}")

# Time-based analysis
time_analysis = analyzer.analyze_by_time_of_day()
print("\n⏰ Trading Activity by Hour:")
for hour, stats in time_analysis.items():
    print(f"  {hour}:00 - Trades: {stats['count']}, Win Rate: {stats['win_rate']:.1f}%")

# Coin performance comparison
coin_comparison = analyzer.compare_coin_performance()
print("\n💰 Best Performing Coins:")
for coin, metrics in coin_comparison[:5]:
    print(f"  {coin}: ${metrics.total_pnl:,.2f} ({metrics.win_rate:.1f}% win rate)")
```

### Whale Bot Intelligence System

```python
from hyperliquid_monitor_plus import (
    WhaleBotCore,
    WhaleBotConfig,
    EnhancedTrade,
    WhaleDetectionEvent
)

# Configure whale bot
config = WhaleBotConfig(
    min_whale_threshold=100_000,  # $100K minimum
    mega_whale_threshold=1_000_000,  # $1M for mega whale
    giant_whale_threshold=5_000_000,  # $5M for giant whale
    max_response_time_ms=100,  # 100ms max processing time
    follow_ratio=0.1,  # Follow with 10% of whale size
    max_position_risk=0.05,  # Max 5% of portfolio per position
    enable_real_time_alerts=True,
    dashboard=False
)

# Create whale bot
whale_bot = WhaleBotCore(config)

# Add detection callback
def on_whale_detected(enhanced_trade: EnhancedTrade):
    print(f"🐋 {enhanced_trade.whale_tier.value} Detected!")
    print(f"   {enhanced_trade.coin}: {enhanced_trade.whale_direction}")
    print(f"   Size: ${enhanced_trade.size_usd:,.0f}")
    print(f"   Urgency: {enhanced_trade.urgency}")
    print(f"   Market Impact: {enhanced_trade.market_impact:.3f}%")
    print(f"   Slippage: {enhanced_trade.slippage:.3f}%")
    print(f"   Follow Opportunity: {enhanced_trade.follow_opportunity}")

whale_bot.add_detection_callback(on_whale_detected)

# Add follow recommendation callback
def on_follow_recommendation(recommendation):
    if recommendation.action == "FOLLOW":
        print(f"✅ FOLLOW RECOMMENDATION")
        print(f"   Confidence: {recommendation.confidence:.1%}")
        print(f"   Reasoning: {recommendation.reasoning}")
        print(f"   Position Size: {recommendation.position_size_ratio:.1%}")
        print(f"   Entry Delay: {recommendation.optimal_entry_delay}s")
        print(f"   Stop Loss: {recommendation.stop_loss_ratio:.1%}")
        print(f"   Take Profit Levels: {recommendation.take_profit_levels}")
    elif recommendation.action == "HEDGE":
        print(f"⚠️ HEDGE RECOMMENDED: {recommendation.reasoning}")
    else:
        print(f"🚫 SKIP: {recommendation.reasoning}")

whale_bot.add_follow_callback(on_follow_recommendation)

# Start whale monitoring
addresses = ["0xwhale1...", "0xwhale2..."]
whale_bot.start_monitoring(addresses, db_path="whale_intelligence.db")

# Get bot statistics
stats = whale_bot.get_bot_statistics()
print(f"Status: {stats['status']}")
print(f"Whales Detected: {stats['performance']['whales_detected']}")
print(f"Follow Recommendations: {stats['performance']['follow_recommendations']}")
print(f"Avg Processing Time: {stats['performance']['avg_processing_time_ms']:.1f}ms")
```

### Report Generation

```python
from hyperliquid_monitor_plus import (
    ReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportType,
    ReportPeriod
)

# Configure report
report_config = ReportConfig(
    report_type=ReportType.PORTFOLIO,
    period=ReportPeriod.WEEKLY,
    format=ReportFormat.HTML,
    include_charts=True,
    include_trades_table=True,
    include_pnl_breakdown=True,
    output_path="reports/"
)

# Create report generator
generator = ReportGenerator(
    db_path="whale_trades.db",
    config=report_config
)

# Generate report
report = generator.generate_report(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)

print(f"Report generated: {report.file_path}")
print(f"Report size: {report.file_size_kb:.1f} KB")
print(f"Trades analyzed: {report.trades_count}")
print(f"Charts generated: {report.charts_count}")

# Generate multiple formats
for fmt in [ReportFormat.HTML, ReportFormat.PDF, ReportFormat.JSON]:
    report_config.format = fmt
    report = generator.generate_report()
    print(f"Generated {fmt.value} report: {report.file_path}")
```

### Notification Integration

```python
from hyperliquid_monitor_plus import (
    NotificationManager,
    TelegramConfig,
    DiscordConfig,
    NotificationPriority
)

# Setup Telegram
telegram_config = TelegramConfig(
    bot_token="YOUR_BOT_TOKEN",
    chat_id="YOUR_CHAT_ID",
    parse_mode="Markdown"
)

# Setup Discord
discord_config = DiscordConfig(
    webhook_url="YOUR_DISCORD_WEBHOOK_URL",
    username="Whale Bot",
    avatar_url="https://example.com/whale-icon.png"
)

# Create notification manager
notif_manager = NotificationManager()
notif_manager.add_telegram(telegram_config)
notif_manager.add_discord(discord_config)

# Send notifications
def on_large_trade(trade):
    message = f"""
🐋 **Large Whale Trade Detected**

**Coin:** {trade.coin}
**Side:** {trade.side}
**Size:** {trade.size:,.4f}
**Price:** ${trade.price:,.2f}
**Value:** ${trade.size * trade.price:,.0f}
**Leverage:** {trade.leverage}x
**Address:** `{trade.address[:10]}...`
    """
    
    # Send to both Telegram and Discord
    notif_manager.send(
        message=message,
        priority=NotificationPriority.HIGH,
        channels=["telegram", "discord"]
    )

# Use with monitor
monitor = HyperliquidMonitor(
    addresses=["0x1234..."],
    callback=on_large_trade
)
```

---

## 🎓 Advanced Features

### Phase 3: ML & Advanced Analytics

```python
from hyperliquid_monitor_plus import (
    WhaleBotCore,
    Phase3Config,
    get_phase3_config
)
import asyncio

# Configure Phase 3 features
phase3_config = get_phase3_config("advanced")
phase3_config.ml_model_enabled = True
phase3_config.cross_chain_enabled = True
phase3_config.dashboard_enabled = True
phase3_config.dashboard_port = 8080
phase3_config.portfolio_optimization_enabled = True

# Create whale bot with Phase 3 capabilities
whale_bot = WhaleBotCore(
    config=whale_bot_config,
    phase3_config=phase3_config
)

# Start advanced features
async def run_advanced_features():
    # Start ML models, cross-chain tracking, and dashboard
    await whale_bot.start_advanced_features()
    
    # Monitor whale activity
    whale_bot.start_monitoring(addresses, "advanced_whale_db.db")
    
    # Run ML prediction for a trade
    prediction = await whale_bot.run_ml_prediction(enhanced_trade)
    if prediction:
        print(f"ML Prediction: {prediction['behavior_score']:.2f}")
        print(f"Price Forecast: ${prediction['predicted_price']:.2f}")
    
    # Track cross-chain activity
    cross_chain_data = await whale_bot.track_cross_chain_activity(enhanced_trade)
    if cross_chain_data:
        print(f"Cross-chain activity detected: {cross_chain_data['chains']}")
    
    # Run portfolio optimization
    optimization = await whale_bot.optimize_portfolio()
    if optimization:
        print("Recommended portfolio adjustments:")
        for coin, recommendation in optimization['adjustments'].items():
            print(f"  {coin}: {recommendation['action']} {recommendation['size']}")
    
    # Get Phase 3 status
    status = await whale_bot.get_phase3_status()
    print(f"ML Models Active: {status['components']['ml_manager']['active_models']}")
    print(f"Cross-chain Whales Tracked: {status['components']['cross_chain_tracker']['whales_tracked']}")
    print(f"Dashboard Running: Port {status['components']['dashboard']['port']}")

# Run the async application
asyncio.run(run_advanced_features())
```

### Cross-Chain Whale Tracking

```python
from hyperliquid_monitor_plus import CrossChainAnalytics, CrossChainMonitor

# Initialize cross-chain analytics
cross_chain = CrossChainAnalytics(
    chains=["ethereum", "arbitrum", "optimism", "base"],
    whale_address="0x1234..."
)

# Track whale across chains
async def track_whale_movements():
    # Get whale activity summary
    summary = await cross_chain.get_whale_summary()
    print(f"Total Value Across Chains: ${summary['total_value_usd']:,.0f}")
    print(f"Active Chains: {summary['active_chains']}")
    
    # Check for coordinated movements
    movements = await cross_chain.detect_coordinated_movements()
    if movements:
        print("⚠️ Coordinated cross-chain activity detected!")
        for movement in movements:
            print(f"  Chain: {movement['chain']}")
            print(f"  Action: {movement['action']}")
            print(f"  Value: ${movement['value_usd']:,.0f}")
            print(f"  Timing: {movement['timing_correlation']:.1%} correlated")

asyncio.run(track_whale_movements())
```

### Real-Time Dashboard

The library includes a beautiful real-time web dashboard for monitoring whale activity:

```python
from hyperliquid_monitor_plus import RealTimeDashboard, AdvancedWhaleBot

# Create whale bot
whale_bot = AdvancedWhaleBot(config)

# Create dashboard
dashboard = RealTimeDashboard(
    whale_bot=whale_bot,
    port=8080,
    host="0.0.0.0"
)

# Start dashboard
async def run_dashboard():
    await dashboard.start_dashboard()
    print("Dashboard running at http://localhost:8080")
    
    # Keep running
    while True:
        await asyncio.sleep(1)

asyncio.run(run_dashboard())
```

Visit `http://localhost:8080` to see:
- Real-time whale trade feed
- Interactive charts and graphs
- Position tracking
- PnL visualization
- Alert notifications
- Market impact analysis

---

## 🛠️ CLI Tools

### Hyperliquid Monitor CLI

```bash
# Basic monitoring
hyperliquid-monitor --address 0x1234... --db whale_trades.db

# Multiple addresses
hyperliquid-monitor --addresses 0x1234... 0x5678... --db trades.db

# With alerts
hyperliquid-monitor --address 0x1234... --alert-discord --db trades.db

# Telegram alerts (requires TELEGRAM_BOT_TOKEN env var)
hyperliquid-monitor --address 0x1234... --alert-telegram

# Silent mode (only database storage)
hyperliquid-monitor --address 0x1234... --db trades.db --silent

# Verbose logging
hyperliquid-monitor --address 0x1234... --db trades.db --verbose

# Disable PnL tracking
hyperliquid-monitor --address 0x1234... --db trades.db --no-pnl

# Dry run (test configuration)
hyperliquid-monitor --address 0x1234... --dry-run
```

### Whale Bot CLI

```bash
# Conservative mode (default)
hyperliquid-whale-bot --address 0x1234... --config config.yaml

# Aggressive mode
hyperliquid-whale-bot --addresses 0x1234... 0x5678... --aggressive-mode

# Demo mode (no real trading)
hyperliquid-whale-bot --address 0x1234... --demo

# Custom parameters
hyperliquid-whale-bot --address 0x1234... \
    --min-whale-size 500000 \
    --max-whales 10 \
    --log-level DEBUG

# Multiple addresses with aggressive strategy
hyperliquid-whale-bot --addresses 0x1234... 0x5678... 0x9abc... \
    --aggressive-mode \
    --min-whale-size 1000000
```

---

## 🛡️ Advanced Risk Management System

Comprehensive risk monitoring and management system for whale tracking.

### Core Risk Classes

```python
from hyperliquid_monitor_plus.risk import (
    AdvancedRiskManager,
    RiskMetrics,
    PositionRisk,
    MarketRisk,
    RiskAlert
)
```

### Key Features

- **Real-time Risk Monitoring**: Continuous assessment of portfolio and position risks
- **Position-specific Analysis**: Individual position risk breakdown
- **Market Regime Detection**: Automatic identification of market conditions
- **Correlation Risk Assessment**: Cross-asset correlation analysis
- **Automated Risk Alerts**: Configurable risk threshold alerts

### Risk Management Example

```python
# Initialize risk manager
risk_manager = AdvancedRiskManager(config)

# Monitor portfolio risk
risk_metrics = await risk_manager.assess_portfolio_risk(positions)
print(f"Portfolio Risk: {risk_metrics.portfolio_risk:.2%}")
print(f"Position Risk: {risk_metrics.position_risk:.2%}")
print(f"Correlation Risk: {risk_metrics.correlation_risk:.2%}")

# Analyze specific position
position_risk = await risk_manager.analyze_position_risk("BTC", position)
if position_risk.risk_level == "HIGH":
    print(f"⚠️ High risk position: {position_risk.symbol}")
    print(f"   Risk Score: {position_risk.total_risk:.2%}")
    print(f"   Var 95%: ${position_risk.var_95_usd:,.2f}")

# Check market conditions
market_risk = await risk_manager.assess_market_conditions()
print(f"Market Regime: {market_risk.market_regime}")
print(f"Volatility Level: {market_risk.volatility_level:.2%}")
print(f"Liquidity: {market_risk.liquidity_conditions}")

# Get risk alerts
alerts = await risk_manager.get_active_risk_alerts()
for alert in alerts:
    print(f"🚨 {alert.severity}: {alert.message}")
    print(f"   Recommended Action: {alert.recommended_action}")
```

---

## 💼 Portfolio Optimization System

AI-powered portfolio optimization with whale intelligence integration.

### Core Optimization Classes

```python
from hyperliquid_monitor_plus.optimization import (
    PortfolioOptimizer,
    RiskAnalyzer,
    WhaleOpportunityDetector,
    PortfolioMetrics,
    OptimizationConstraints
)
```

### Key Features

- **AI-powered Portfolio Rebalancing**: Intelligent allocation optimization
- **Whale Trading Opportunity Detection**: Identify follow/hedge opportunities
- **Risk-adjusted Optimization**: VaR and volatility constrained optimization
- **Correlation-aware Allocation**: Multi-asset correlation consideration
- **Real-time Portfolio Monitoring**: Live portfolio performance tracking

### Portfolio Optimization Example

```python
# Initialize portfolio optimizer
optimizer = PortfolioOptimizer(config)

# Optimize current portfolio
optimization_result = await optimizer.optimize_portfolio(positions)
print("📊 Portfolio Optimization Results:")
print(f"Expected Return: {optimization_result.expected_return:.2%}")
print(f"Portfolio Risk: {optimization_result.portfolio_risk:.2%}")
print(f"Sharpe Ratio: {optimization_result.sharpe_ratio:.2f}")

print("\n📈 Recommended Allocation:")
for asset, weight in optimization_result.recommended_weights.items():
    current_weight = positions.get(asset, {}).get('weight', 0)
    change = weight - current_weight
    print(f"  {asset}: {weight:.2%} (change: {change:+.2%})")

# Detect whale opportunities
opportunities = await optimizer.opportunity_detector.detect_follow_opportunities(whale_trades)
for opp in opportunities:
    print(f"\n🐋 Whale Opportunity Detected:")
    print(f"   Asset: {opp.asset}")
    print(f"   Type: {opp.opportunity_type}")
    print(f"   Expected Return: {opp.expected_return:.2%}")
    print(f"   Confidence: {opp.confidence:.1%}")
    print(f"   Suggested Allocation: {opp.suggested_allocation:.2%}")
    print(f"   Entry Timing: {opp.entry_timing}")

# Get portfolio metrics
metrics = optimizer.risk_analyzer.calculate_portfolio_risk(positions)
print(f"\n📊 Portfolio Metrics:")
print(f"   Total Value: ${metrics.total_value_usd:,.2f}")
print(f"   Daily Return: {metrics.daily_return:.2%}")
print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
print(f"   VaR 95%: ${metrics.var_95:,.2f}")
```

---

## ⚙️ Advanced Configuration Management

Comprehensive configuration system for all library features.

### Configuration Classes

```python
from hyperliquid_monitor_plus.config import (
    Config,
    Phase3Config,
    WhaleBotConfig
)
```

### Configuration Methods

#### Environment Variables

```bash
# Basic API Configuration
export HYPERLIQUID_API_URL="https://api.hyperliquid.xyz"
export HYPERLIQUID_WS_URL="wss://ws.hyperliquid.xyz"
export DATABASE_PATH="whale_trades.db"

# Risk Management Settings
export MAX_POSITION_SIZE=0.25
export MAX_PORTFOLIO_RISK=0.15
export RISK_WARNING_THRESHOLD=0.10
export CRITICAL_RISK_THRESHOLD=0.20

# Portfolio Optimization
export ENABLE_PORTFOLIO_OPTIMIZATION=true
export OPTIMIZATION_INTERVAL_HOURS=1
export MAX_WHALE_CORRELATION=0.7
export REBALANCE_THRESHOLD=0.05

# ML and Intelligence
export ML_MODEL_PATH="/models/"
export ENABLE_SENTIMENT_ANALYSIS=true
export CROSS_CHAIN_ENABLED=true
export SENTIMENT_UPDATE_INTERVAL=300

# Notifications
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

#### Configuration Files (YAML)

```yaml
# config.yaml - Complete Configuration

# Core Whale Bot Settings
whale_bot:
  min_whale_threshold: 100000      # $100K minimum
  mega_whale_threshold: 1000000    # $1M for mega whale
  giant_whale_threshold: 5000000   # $5M for giant whale
  max_response_time_ms: 100
  max_whales: 10
  follow_ratio: 0.1               # Follow with 10% of whale size
  max_position_risk: 0.05          # Max 5% risk per position
  enable_real_time_alerts: true
  alert_cooldown_seconds: 30

# Advanced Risk Management
risk_management:
  enabled: true
  max_position_size: 0.25          # 25% max per position
  max_portfolio_risk: 0.15         # 15% max portfolio risk
  high_risk_threshold: 0.10        # Warning at 10%
  critical_risk_threshold: 0.20    # Critical at 20%
  auto_position_management: true
  liquidation_buffer: 0.05         # 5% buffer from liquidation

# Portfolio Optimization
portfolio_optimization:
  enabled: true
  optimization_frequency: "hourly"  # hourly, daily, weekly
  risk_budget: 0.15                # Risk allocation
  correlation_limit: 0.7           # Max correlation between assets
  rebalance_threshold: 0.05        # 5% drift triggers rebalance
  min_trade_size_usd: 1000
  max_turnover: 0.20               # 20% max turnover

# Phase 3 Advanced Features
phase3:
  ml_model_enabled: true
  ml_model_config:
    model_path: "/models/whale_behavior_v3.pkl"
    confidence_threshold: 0.75
    auto_update: true
    update_frequency: "daily"
  
  cross_chain_enabled: true
  cross_chain_config:
    enabled_chains: ["ethereum", "arbitrum", "optimism", "base"]
    sync_frequency_seconds: 30
    enable_coordination_detection: true
  
  dashboard_enabled: true
  dashboard_config:
    port: 8080
    host: "0.0.0.0"
    enable_api: true
    enable_websockets: true
    auth_enabled: false
  
  portfolio_optimization_enabled: true
  optimization_config:
    risk_model: "historical"
    optimization_algorithm: "mean_variance"
    rebalance_frequency: "weekly"
  
  sentiment_analysis_enabled: true
  sentiment_config:
    data_sources: ["twitter", "reddit", "discord"]
    update_interval: 300           # 5 minutes
    sentiment_threshold: 0.1       # Alert threshold

# Notification Channels
notifications:
  discord:
    enabled: true
    webhook_url: "${DISCORD_WEBHOOK_URL}"
    username: "Whale Bot"
    avatar_url: "https://example.com/whale-icon.png"
    alert_levels: ["WARNING", "CRITICAL"]
  
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"
    parse_mode: "Markdown"
    alert_levels: ["INFO", "WARNING", "CRITICAL"]
  
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your_email@gmail.com"
    password: "your_app_password"
    recipients: ["admin@yourcompany.com"]

# Database Configuration
database:
  path: "whale_trades.db"
  backup_enabled: true
  backup_interval_hours: 24
  retention_days: 365
  vacuum_interval_days: 7

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "logs/whale_monitor.log"
  max_file_size_mb: 100
  backup_count: 5
  enable_console: true
```

### Configuration Loading Example

```python
# Load configuration from multiple sources
config = Config()

# Load from environment variables
config.load_from_env()

# Load from YAML file
config.load_from_file("config.yaml")

# Load from JSON file
config.load_from_file("config.json")

# Create specific configurations
whale_config = WhaleBotConfig(
    min_whale_threshold=100_000,
    mega_whale_threshold=1_000_000,
    giant_whale_threshold=5_000_000,
    follow_ratio=0.1
)

phase3_config = Phase3Config(
    ml_model_enabled=True,
    cross_chain_enabled=True,
    dashboard_enabled=True,
    dashboard_port=8080
)

# Validate configuration
validation_result = config.validate()
if not validation_result.is_valid:
    print("Configuration errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

---

## 📊 Advanced Type System Reference

Comprehensive type definitions for all library features.

### Enhanced Trade Types

```python
from hyperliquid_monitor_plus.core_types.types import (
    EnhancedTrade,
    WhaleTier,
    WhaleDetectionEvent,
    MarketImpactAnalysis,
    TradeUrgency
)

# Trade Urgency Levels
class TradeUrgency(str, Enum):
    LOW = "LOW"        # Normal trading activity
    MEDIUM = "MEDIUM"  # Elevated activity
    HIGH = "HIGH"      # Unusual activity
    CRITICAL = "CRITICAL"  # Emergency/liquidation

# Whale Tier Classification
whale_tier = EnhancedTrade(
    coin="BTC",
    size=100.5,
    price=45000,
    whale_tier=WhaleTier.MEGA_WHALE,
    urgency=TradeUrgency.HIGH,
    market_impact=0.0025,  # 0.25% price impact
    follow_opportunity=True
)
```

### Strategy & Risk Types

```python
from hyperliquid_monitor_plus.core_types.strategy_types import (
    StrategyMetrics,
    RiskAssessment,
    PositionRisk,
    MarketRisk,
    PatternType,
    RecommendationType
)

# Strategy Performance Metrics
metrics = StrategyMetrics(
    total_trades=150,
    win_rate=0.68,
    profit_factor=1.45,
    sharpe_ratio=1.23,
    max_drawdown=0.08,
    avg_trade_duration=timedelta(hours=4.5),
    best_trade_pnl=5000.0,
    worst_trade_pnl=-1200.0
)

# Risk Assessment Types
risk_assessment = RiskAssessment(
    overall_risk_score=0.65,      # 0-1 scale
    position_risk_score=0.45,
    correlation_risk_score=0.78,
    liquidity_risk_score=0.23,
    volatility_risk_score=0.56,
    recommended_actions=[
        "Reduce BTC exposure by 5%",
        "Increase stable coin allocation",
        "Set tighter stop losses"
    ]
)
```

### Portfolio & Optimization Types

```python
from hyperliquid_monitor_plus.optimization.portfolio_optimizer import (
    PortfolioMetrics,
    OptimizationConstraints,
    WhaleOpportunity,
    PortfolioPosition
)

# Portfolio Position
position = PortfolioPosition(
    asset="BTC",
    current_weight=0.25,
    target_weight=0.20,
    market_value_usd=25000,
    entry_price=44000,
    current_price=45000,
    unrealized_pnl=1000,
    whale_exposure=True,
    risk_contribution=0.15
)

# Whale Trading Opportunity
opportunity = WhaleOpportunity(
    opportunity_id="btc_follow_001",
    asset="BTC",
    opportunity_type="FOLLOW",
    expected_return=0.035,        # 3.5% expected return
    confidence=0.78,              # 78% confidence
    risk_level="MEDIUM",
    whale_address="0x1234...",
    suggested_allocation=0.05,    # 5% of portfolio
    entry_timing="immediate"      # immediate, 5min, 15min, 1hour
)
```

### Alert & Notification Types

```python
from hyperliquid_monitor_plus.risk.risk_manager import RiskAlert
from hyperliquid_monitor_plus.alerts.types import AlertType, AlertLevel

# Risk Alert
risk_alert = RiskAlert(
    alert_type="HIGH_RISK_POSITION",
    severity="WARNING",
    message="BTC position approaching liquidation threshold",
    recommended_action="Reduce position size or add margin",
    position_details={
        "symbol": "BTC",
        "size": 2.5,
        "leverage": 10,
        "distance_to_liquidation": 0.08  # 8% away
    }
)

# Alert Condition Types
class AlertType(str, Enum):
    LARGE_TRADE = "LARGE_TRADE"
    WHALE_ACTIVITY = "WHALE_ACTIVITY"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    LIQUIDATION = "LIQUIDATION"
    RISK_ALERT = "RISK_ALERT"
    PORTFOLIO_ALERT = "PORTFOLIO_ALERT"

class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"
```

### Dashboard & Visualization Types

```python
from hyperliquid_monitor_plus.visualization.dashboard_components import (
    TradingMetrics,
    AlertData,
    ChartConfig
)

# Real-time Trading Metrics
metrics = TradingMetrics(
    timestamp=datetime.now(),
    total_volume_24h=1_250_000_000,
    whale_volume_24h=250_000_000,
    active_whales=15,
    avg_trade_size=125_000,
    largest_trade_size=5_000_000,
    volume_change_24h=0.15,  # 15% increase
    whale_activity_score=0.78
)

# Chart Configuration
chart_config = ChartConfig(
    chart_type="line",
    timeframe="1h",
    indicators=["SMA", "RSI", "Volume"],
    colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    width=1200,
    height=600,
    show_grid=True,
    show_legend=True
)
```

---

## 🎨 Advanced Visualization System

Comprehensive visualization and dashboard system with real-time updates.

### Chart Classes

```python
from hyperliquid_monitor_plus.visualization import (
    WhaleVolumeChart,
    MarketImpactPlot,
    TradeDistributionChart,
    PortfolioPerformanceChart,
    SentimentAnalysisPlot,
    CorrelationHeatmap,
    RealTimeWhaleTracker,
    
    # Dashboard Components
    TradingMetricsWidget,
    AlertDistributionWidget,
    PerformanceDashboard,
    RealTimeChart,
    
    # Exporters
    ChartExporter,
    HTMLReportGenerator,
    PDFReportGenerator
)
```

### Key Features

- **Interactive Charts**: Plotly-based interactive visualizations
- **Real-time Updates**: Live data streaming and chart updates
- **Multiple Export Formats**: PNG, PDF, SVG, HTML export
- **Dashboard Components**: Reusable dashboard widgets
- **Custom Styling**: Configurable themes and styling

### Visualization Examples

```python
# Create interactive whale volume chart
chart = WhaleVolumeChart(figsize=(14, 10))
chart.plot_whale_volume_tracking(trade_data, time_range="7d")
chart.add_market_impact_overlay()
chart.set_title("Whale Activity Analysis - Last 7 Days")

# Market impact analysis
impact_plot = MarketImpactPlot(figsize=(16, 12))
impact_plot.plot_market_impact_analysis(whale_trades)
impact_plot.add_price_correlation_lines()
impact_plot.add_volatility_bands()

# Portfolio performance chart
portfolio_chart = PortfolioPerformanceChart(figsize=(14, 8))
portfolio_chart.plot_portfolio_performance(positions, period="1y")
portfolio_chart.add_benchmark_comparison("BTC")
portfolio_chart.add_drawdown_periods()

# Real-time whale tracker
tracker = RealTimeWhaleTracker(update_interval=5000)  # 5 seconds
tracker.start_real_time_tracking(whale_addresses)

# Dashboard widgets
metrics_widget = TradingMetricsWidget(update_interval=10000)
alert_widget = AlertDistributionWidget()
performance_dashboard = PerformanceDashboard()

# Export charts
exporter = ChartExporter()
exporter.export_chart(chart, "whale_analysis.png", format="png", dpi=300)
exporter.export_chart(impact_plot, "market_impact.pdf", format="pdf")

# Generate comprehensive HTML report
html_generator = HTMLReportGenerator()
report_path = html_generator.generate_comprehensive_report(
    trade_data=trade_data,
    whale_addresses=whale_addresses,
    period="30d",
    include_charts=True,
    include_metrics=True,
    output_path="reports/monthly_report.html"
)

print(f"Report generated: {report_path}")
```

### Advanced Dashboard Integration

```python
from hyperliquid_monitor_plus.visualization.dashboard_components import (
    RealTimeDashboard,
    DashboardIntegration
)

# Create comprehensive dashboard
dashboard = RealTimeDashboard(
    whale_bot=whale_bot,
    port=8080,
    host="0.0.0.0",
    enable_api=True,
    enable_websockets=True,
    theme="dark"
)

# Add custom widgets
dashboard.add_widget(TradingMetricsWidget(), position=(0, 0), size=(6, 4))
dashboard.add_widget(RealTimeChart(chart_type="whale_volume"), position=(6, 0), size=(6, 4))
dashboard.add_widget(AlertDistributionWidget(), position=(0, 4), size=(12, 4))
dashboard.add_widget(PerformanceDashboard(), position=(0, 8), size=(12, 4))

# Start dashboard
async def run_dashboard():
    await dashboard.start_dashboard()
    print("🌐 Dashboard running at http://localhost:8080")
    print("📊 Available endpoints:")
    print("   /metrics - Trading metrics API")
    print("   /trades - Real-time trade stream")
    print("   /alerts - Alert management")
    print("   /portfolio - Portfolio analytics")
    
    # Keep running
    while True:
        await asyncio.sleep(1)

# Access via web browser or API
dashboard_url = "http://localhost:8080"
api_endpoints = [
    f"{dashboard_url}/api/v1/metrics",
    f"{dashboard_url}/api/v1/trades",
    f"{dashboard_url}/api/v1/whales",
    f"{dashboard_url}/api/v1/alerts",
    f"{dashboard_url}/api/v1/portfolio"
]
```

---

## 🧠 ML Integration & Advanced Intelligence

Machine learning powered whale intelligence system with social sentiment analysis.

### Intelligence Classes

```python
from hyperliquid_monitor_plus.intelligence import (
    WhaleIntelligence,
    SocialSentimentAnalyzer,
    IntelligenceEnhancements,
    MLModelManager
)
```

### Key Features

- **Social Sentiment Analysis**: Real-time sentiment from social media
- **Pattern Recognition**: ML-powered behavior pattern detection
- **Predictive Analytics**: Whale behavior prediction models
- **Market Regime Detection**: Automatic market condition classification
- **Cross-chain Intelligence**: Multi-blockchain activity analysis

### ML Intelligence Examples

```python
# Initialize intelligence system
intelligence = WhaleIntelligence(config)

# Analyze trade with ML enhancement
enhanced_trade = await intelligence.analyze_trade_with_ml(trade)
print(f"🤖 ML Analysis Results:")
print(f"   Behavior Score: {enhanced_trade.ml_prediction.behavior_score:.2f}")
print(f"   Confidence: {enhanced_trade.ml_prediction.confidence:.1%}")
print(f"   Pattern Match: {enhanced_trade.ml_prediction.pattern_type}")
print(f"   Predicted Follow Success: {enhanced_trade.ml_prediction.follow_success_rate:.1%}")

# Social sentiment analysis
sentiment_analyzer = SocialSentimentAnalyzer()
sentiment_score = await sentiment_analyzer.analyze_sentiment(
    symbol="BTC",
    timeframe="1h",
    data_sources=["twitter", "reddit", "discord"]
)
print(f"📱 Social Sentiment Analysis:")
print(f"   Overall Score: {sentiment_score.overall_score:.2f}")
print(f"   Positive: {sentiment_score.positive:.1%}")
print(f"   Negative: {sentiment_score.negative:.1%}")
print(f"   Neutral: {sentiment_score.neutral:.1%}")
print(f"   Volume: {sentiment_score.volume_score:.2f}")
print(f"   Trend: {sentiment_score.trend}")

# Market regime detection
regime_analysis = await intelligence.detect_market_regime(
    price_data=recent_prices,
    volume_data=recent_volumes,
    whale_activity=whale_trades
)
print(f"📈 Market Regime Detection:")
print(f"   Current Regime: {regime_analysis.regime}")
print(f"   Confidence: {regime_analysis.confidence:.1%}")
print(f"   Volatility Level: {regime_analysis.volatility_level}")
print(f"   Liquidity Condition: {regime_analysis.liquidity_condition}")
print(f"   Recommended Strategy: {regime_analysis.recommended_strategy}")

# Cross-chain intelligence
cross_chain_data = await intelligence.analyze_cross_chain_activity(
    whale_address="0x1234...",
    time_window="24h"
)
if cross_chain_data:
    print(f"🔗 Cross-chain Analysis:")
    print(f"   Active Chains: {cross_chain_data.active_chains}")
    print(f"   Total Value: ${cross_chain_data.total_value_usd:,.2f}")
    print(f"   Coordination Score: {cross_chain_data.coordination_score:.2f}")
    print(f"   Suspicious Activity: {cross_chain_data.suspicious_patterns}")
```

### Advanced ML Model Management

```python
# ML Model Manager
ml_manager = MLModelManager(config)

# Load and manage models
models = await ml_manager.load_models([
    "whale_behavior_model_v3.pkl",
    "market_regime_classifier.pkl",
    "sentiment_analyzer.pkl"
])

# Update model with new data
await ml_manager.update_model(
    model_name="whale_behavior_model_v3",
    new_training_data=recent_whale_trades,
    validation_split=0.2
)

# Get model performance metrics
performance = await ml_manager.get_model_performance("whale_behavior_model_v3")
print(f"🎯 Model Performance:")
print(f"   Accuracy: {performance.accuracy:.2%}")
print(f"   Precision: {performance.precision:.2%}")
print(f"   Recall: {performance.recall:.2%}")
print(f"   F1-Score: {performance.f1_score:.2%}")
print(f"   Last Updated: {performance.last_updated}")
```

---

## 🔗 Integration & External Services

Seamless integration with external services and platforms.

### Dashboard Integration

```python
from hyperliquid_monitor_plus.integration import (
    RealTimeDashboard,
    DashboardIntegration,
    ExternalAPI
)
```

### Integration Examples

```python
# Create dashboard integration
integration = DashboardIntegration(
    whale_bot=whale_bot,
    port=8080,
    host="0.0.0.0",
    enable_api=True,
    enable_websockets=True,
    enable_auth=False
)

# Add webhook endpoints
integration.add_webhook_endpoint(
    path="/whale-alert",
    methods=["POST"],
    callback=external_alert_handler,
    auth_required=False
)

integration.add_webhook_endpoint(
    path="/portfolio-update",
    methods=["POST"],
    callback=portfolio_update_handler,
    auth_required=True,
    api_key="your_api_key"
)

# External API integration
external_api = ExternalAPI(
    base_url="https://api.yourplatform.com",
    api_key="your_api_key",
    timeout=30
)

# Subscribe to external data feeds
await integration.subscribe_to_price_feed(
    symbols=["BTC", "ETH", "SOL"],
    callback=on_price_update
)

await integration.subscribe_to_social_sentiment(
    platforms=["twitter", "reddit"],
    callback=on_sentiment_update
)

# Start integration service
async def run_integration():
    await integration.start_integration()
    print("🔗 Integration services running:")
    print("   WebSocket: ws://localhost:8080/ws")
    print("   REST API: http://localhost:8080/api/v1/")
    print("   Webhooks: http://localhost:8080/webhook/")
    
    # Monitor integration health
    while True:
        health = await integration.get_health_status()
        if not health.is_healthy:
            print(f"⚠️ Integration issue: {health.error}")
        await asyncio.sleep(60)

# External service callbacks
async def external_alert_handler(data):
    """Handle external webhook alerts"""
    alert_type = data.get('type')
    message = data.get('message')
    
    # Process external alert
    if alert_type == 'whale_detected':
        await process_external_whale_alert(message)
    elif alert_type == 'risk_alert':
        await process_external_risk_alert(message)

async def portfolio_update_handler(data):
    """Handle portfolio updates from external system"""
    positions = data.get('positions', [])
    await whale_bot.update_external_positions(positions)
```

### Third-party Service Integration

```python
# TradingView integration
from hyperliquid_monitor_plus.integration.tradingview import TradingViewConnector

tv_connector = TradingViewConnector(
    webhook_secret="your_webhook_secret",
    chart_template="whale_analysis"
)

await tv_connector.send_chart_update(
    symbol="BTCUSDT",
    timeframe="1h",
    chart_data=chart_data
)

# Trading platform integration
from hyperliquid_monitor_plus.integration.broker import BrokerConnector

broker_connector = BrokerConnector(
    broker_type="binance",
    api_key="your_api_key",
    secret_key="your_secret_key"
)

# Execute trades based on whale signals
async def execute_whale_follow_signal(signal):
    if signal.action == "FOLLOW":
        order = await broker_connector.create_order(
            symbol=signal.symbol,
            side="BUY",
            quantity=signal.quantity,
            order_type="MARKET"
        )
        print(f"✅ Follow order executed: {order.order_id}")
```

---

## 📁 Examples Directory & Usage Guide

Complete examples for all library features with detailed documentation.

### Directory Structure

```
examples/
├── README.md                          # This guide
├── run_examples.py                    # Master script to run all examples
│
├── core/                              # Core whale bot examples
│   └── whale_bot_example.py          # Full whale bot implementation
│
├── monitoring/                        # Basic monitoring examples
│   └── basic_monitoring.py           # Simple monitoring setup
│
├── analysis/                          # Market analysis examples
│   └── market_analysis.py            # Market impact & correlation analysis
│
├── intelligence/                      # ML and AI examples
│   └── intelligence_demo.py          # Social sentiment & ML integration
│
├── tracking/                          # Advanced tracking examples
│   └── advanced_tracking.py          # Cross-chain & real-time tracking
│
├── configuration/                     # Configuration examples
│   └── advanced_config.py            # Complex configuration management
│
├── database/                          # Database examples
│   └── data_analysis.py              # Database queries & analytics
│
├── integration/                       # Integration examples
│   └── dashboard_integration.py      # Dashboard & API integration
│
├── notifications/                     # Alert examples
│   └── alert_system.py               # Multi-channel notification system
│
├── strategy/                          # Strategy analysis examples
│   └── strategy_analyzer.py          # Strategy performance analysis
│
└── visualization/                     # Chart and visualization examples
    ├── simple_demo.py                # Basic chart creation
    └── comprehensive_demo.py         # Advanced visualization suite
```

### Running Examples

#### Run All Examples

```bash
# Run all examples in sequence
python examples/run_examples.py

# Run with specific configuration
python examples/run_examples.py --config advanced_config.yaml --output reports/

# Run only specific modules
python examples/run_examples.py --modules core,monitoring,analysis
```

#### Run Individual Examples

```bash
# Core whale bot
python examples/core/whale_bot_example.py \
    --addresses 0x1234... 0x5678... \
    --config config.yaml \
    --output whale_bot.log

# Basic monitoring
python examples/monitoring/basic_monitoring.py \
    --address 0x1234... \
    --db trades.db \
    --alert-discord

# Market analysis
python examples/analysis/market_analysis.py \
    --db trades.db \
    --period 30d \
    --charts \
    --output analysis_report.html

# Intelligence demo
python examples/intelligence/intelligence_demo.py \
    --ml-enabled \
    --sentiment-enabled \
    --cross-chain-enabled

# Advanced tracking
python examples/tracking/advanced_tracking.py \
    --addresses addr1 addr2 addr3 \
    --dashboard-enabled \
    --ml-predictions \
    --real-time

# Configuration examples
python examples/configuration/advanced_config.py \
    --generate-config \
    --validate-config \
    --export-examples

# Database analysis
python examples/database/data_analysis.py \
    --db trades.db \
    --export-csv \
    --generate-stats

# Dashboard integration
python examples/integration/dashboard_integration.py \
    --port 8080 \
    --enable-websockets \
    --api-key your_key

# Notification system
python examples/notifications/alert_system.py \
    --discord-webhook $DISCORD_WEBHOOK \
    --telegram-token $TELEGRAM_TOKEN \
    --test-alerts

# Strategy analyzer
python examples/strategy/strategy_analyzer.py \
    --address 0x1234... \
    --period 90d \
    --export-report

# Visualization demos
python examples/visualization/simple_demo.py
python examples/visualization/comprehensive_demo.py \
    --export-pdf \
    --export-html \
    --real-time
```

### Example Customization

#### Create Custom Example

```python
#!/usr/bin/env python3
"""
Custom Whale Monitoring Example
Demonstrates custom implementation using hyperliquid-monitor-plus
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from hyperliquid_monitor_plus import (
    HyperliquidMonitor,
    AlertManager,
    PortfolioOptimizer,
    AdvancedRiskManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomWhaleMonitor:
    """Custom whale monitoring implementation"""
    
    def __init__(self, config):
        self.config = config
        self.monitor = None
        self.alert_manager = AlertManager()
        self.risk_manager = AdvancedRiskManager(config)
        self.portfolio_optimizer = PortfolioOptimizer(config)
        
    async def setup_alerts(self):
        """Setup custom alert conditions"""
        # Add custom alert for large BTC trades
        await self.alert_manager.add_condition(
            AlertCondition(
                name="Major BTC Movement",
                alert_type=AlertType.LARGE_TRADE,
                level=AlertLevel.CRITICAL,
                min_volume_usd=2_000_000,
                coins=["BTC"],
                message_template="🚨 Major BTC movement: ${volume_usd:,.0f}"
            )
        )
        
    async def on_whale_detected(self, trade):
        """Custom whale detection handler"""
        logger.info(f"🐋 Whale detected: {trade.coin} {trade.side}")
        
        # Analyze with ML
        enhanced_trade = await self.enhance_trade_with_ml(trade)
        
        # Check risk
        risk_score = await self.risk_manager.assess_trade_risk(enhanced_trade)
        
        # Generate portfolio recommendation
        if enhanced_trade.follow_opportunity:
            recommendation = await self.portfolio_optimizer.analyze_opportunity(
                enhanced_trade
            )
            
            if recommendation.confidence > 0.8:
                logger.info(f"✅ High-confidence follow opportunity: {recommendation}")
                
    async def enhance_trade_with_ml(self, trade):
        """Enhance trade with ML analysis"""
        # Add ML predictions here
        # This is a simplified example
        enhanced_trade = trade.copy()
        enhanced_trade.ml_prediction = {
            'behavior_score': 0.85,
            'confidence': 0.78,
            'pattern_type': 'accumulation',
            'follow_success_rate': 0.72
        }
        return enhanced_trade
        
    async def start_monitoring(self, addresses: List[str]):
        """Start the custom monitoring system"""
        await self.setup_alerts()
        
        self.monitor = HyperliquidMonitor(
            addresses=addresses,
            db_path="custom_whale.db",
            callback=self.on_whale_detected,
            alert_callback=self.alert_manager.check_trade,
            track_pnl=True
        )
        
        logger.info("🚀 Starting custom whale monitor...")
        await self.monitor.start()

async def main():
    """Main execution function"""
    # Configuration
    addresses = [
        "0x1234567890123456789012345678901234567890",
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    ]
    
    # Initialize and start monitor
    monitor = CustomWhaleMonitor(config)
    await monitor.start_monitoring(addresses)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🚨 Advanced Alert & Notification System

Comprehensive alert and notification system with multiple channels and advanced filtering.

### Alert Classes

```python
from hyperliquid_monitor_plus.alerts import (
    AlertManager,
    AlertCondition,
    AlertLevel,
    AlertType,
    TradeFilter,
    AlertFilter
)

from hyperliquid_monitor_plus.notifications import (
    NotificationManager,
    TelegramConfig,
    DiscordConfig,
    NotificationPriority
)
```

### Advanced Alert Features

- **Multi-level Alerting**: INFO, WARNING, CRITICAL, EMERGENCY
- **Smart Filtering**: Advanced trade and address filtering
- **Channel Management**: Multiple notification channels
- **Alert History**: Persistent alert tracking and analysis
- **Escalation Rules**: Automatic alert escalation
- **Alert Correlation**: Intelligent alert grouping

### Advanced Alert Examples

```python
# Setup comprehensive alert system
alert_manager = AlertManager()

# 1. Large Trade Alert with Geographic Filter
alert_manager.add_condition(
    AlertCondition(
        name="Major Altcoin Alert",
        alert_type=AlertType.LARGE_TRADE,
        level=AlertLevel.WARNING,
        min_volume_usd=500_000,
        coins=["SOL", "AVAX", "ARB", "OP", "MATIC"],
        exclude_addresses=["0xexcluded1...", "0xexcluded2..."],
        message_template="🔔 Large {side} on {coin}: {size} @ ${price} (${volume_usd:,.0f})",
        cooldown_seconds=300  # 5-minute cooldown
    )
)

# 2. Whale Activity Pattern Detection
alert_manager.add_condition(
    AlertCondition(
        name="Whale Pattern Detection",
        alert_type=AlertType.WHALE_ACTIVITY,
        level=AlertLevel.CRITICAL,
        addresses=["0xknownwhale1...", "0xknownwhale2..."],
        min_volume_usd=1_000_000,
        pattern_detection=True,
        correlation_threshold=0.8,
        message_template="🐋 Known whale {address} pattern detected: ${volume_usd:,.0f}"
    )
)

# 3. Liquidation Risk Alert
alert_manager.add_condition(
    AlertCondition(
        name="Liquidation Risk Monitor",
        alert_type=AlertType.LIQUIDATION,
        level=AlertLevel.EMERGENCY,
        min_volume_usd=2_000_000,
        sides=["SELL"],
        liquidation_probability_threshold=0.7,
        message_template="⚡ HIGH LIQUIDATION RISK: {coin} - ${volume_usd:,.0f}"
    )
)

# 4. Portfolio Risk Alert
alert_manager.add_condition(
    AlertCondition(
        name="Portfolio Risk Monitor",
        alert_type=AlertType.PORTFOLIO_ALERT,
        level=AlertLevel.CRITICAL,
        min_portfolio_risk=0.15,  # 15% portfolio risk
        risk_increase_threshold=0.05,  # 5% increase
        message_template="⚠️ Portfolio risk elevated: {risk_level:.1%}"
    )
)

# Advanced Trade Filtering
trade_filter = TradeFilter()
trade_filter.add_condition("min_volume_usd", 100_000)
trade_filter.add_condition("coins", ["BTC", "ETH"])
trade_filter.add_condition("exclude_addresses", ["0xspam1...", "0xspam2..."])
trade_filter.add_condition("time_range", {
    "start": datetime.now() - timedelta(hours=1),
    "end": datetime.now()
})

# Alert Correlation and Escalation
alert_correlation = {
    "time_window": 300,  # 5 minutes
    "similarity_threshold": 0.7,
    "escalation_rules": [
        {"condition": "3_alerts_in_5min", "action": "increase_level"},
        {"condition": "same_whale_multiple_trades", "action": "combine_alerts"}
    ]
}

# Get alert analytics
analytics = alert_manager.get_analytics(time_range="7d")
print(f"📊 Alert Analytics (7 days):")
print(f"   Total Alerts: {analytics.total_alerts}")
print(f"   Critical Alerts: {analytics.critical_count}")
print(f"   False Positives: {analytics.false_positive_rate:.1%}")
print(f"   Average Response Time: {analytics.avg_response_time:.1f}s")
print(f"   Most Active Coin: {analytics.most_active_coin}")
```

### Multi-Channel Notification System

```python
# Setup notification manager
notification_manager = NotificationManager()

# Discord Configuration
discord_config = DiscordConfig(
    webhook_url="https://discord.com/api/webhooks/...",
    username="🐋 Whale Monitor",
    avatar_url="https://example.com/whale-icon.png",
    embed_color=0x0099ff,
    mention_everyone=False
)

# Telegram Configuration
telegram_config = TelegramConfig(
    bot_token="your_bot_token",
    chat_id="your_chat_id",
    parse_mode="Markdown",
    disable_web_page_preview=True
)

# Email Configuration
email_config = EmailConfig(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your_email@gmail.com",
    password="your_app_password",
    recipients=["admin@company.com", "trader@company.com"],
    use_tls=True
)

# Slack Configuration
slack_config = SlackConfig(
    webhook_url="https://hooks.slack.com/...",
    channel="#whale-alerts",
    username="Whale Bot",
    icon_emoji=":whale:"
)

# Add notification channels
notification_manager.add_discord(discord_config)
notification_manager.add_telegram(telegram_config)
notification_manager.add_email(email_config)
notification_manager.add_slack(slack_config)

# Create rich notification messages
async def create_rich_whale_alert(trade):
    """Create comprehensive whale alert"""
    alert_message = f"""
🐋 **WHALE ALERT DETECTED**

**Trade Details:**
• **Coin:** {trade.coin}
• **Side:** {trade.side}
• **Size:** {trade.size:,.4f}
• **Price:** ${trade.price:,.2f}
• **Value:** ${trade.size * trade.price:,.0f}
• **Address:** `{trade.address[:10]}...`

**Market Impact:**
• **Estimated Impact:** {trade.estimated_impact:.3f}%
• **Slippage:** {trade.estimated_slippage:.3f}%
• **Follow Opportunity:** {'✅ Yes' if trade.follow_opportunity else '❌ No'}

**Alert Level:** {get_alert_level(trade)}
**Timestamp:** {trade.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
    """
    
    return alert_message

# Send alerts with different priorities
async def send_whale_alert(trade):
    """Send comprehensive whale alert"""
    message = await create_rich_whale_alert(trade)
    
    # Send to all channels
    await notification_manager.send(
        message=message,
        priority=NotificationPriority.HIGH,
        channels=["discord", "telegram", "slack"],
        format_message=True
    )
    
    # Send urgent alerts via email
    if trade.size * trade.price > 5_000_000:  # $5M+ trades
        await notification_manager.send(
            message=message,
            priority=NotificationPriority.EMERGENCY,
            channels=["email"],
            format_message=True
        )

# Alert Management Dashboard
class AlertDashboard:
    """Real-time alert management dashboard"""
    
    def __init__(self, alert_manager, notification_manager):
        self.alert_manager = alert_manager
        self.notification_manager = notification_manager
        
    async def get_alert_dashboard(self):
        """Get real-time alert dashboard data"""
        dashboard_data = {
            "active_alerts": await self.alert_manager.get_active_alerts(),
            "recent_alerts": await self.alert_manager.get_recent_alerts(limit=10),
            "alert_statistics": await self.alert_manager.get_statistics(),
            "notification_status": await self.notification_manager.get_status(),
            "alert_rules": await self.alert_manager.list_conditions()
        }
        return dashboard_data
        
    async def enable_alert_rule(self, rule_name: str):
        """Enable specific alert rule"""
        return await self.alert_manager.enable_condition(rule_name)
        
    async def disable_alert_rule(self, rule_name: str):
        """Disable specific alert rule"""
        return await self.alert_manager.disable_condition(rule_name)
        
    async def test_alert_rule(self, rule_name: str):
        """Test alert rule with sample data"""
        test_trade = create_test_trade()
        result = await self.alert_manager.test_condition(rule_name, test_trade)
        return result
```

---

## 📈 Performance Monitoring & Optimization

Built-in performance monitoring and optimization tools for production environments.

### Performance Classes

```python
from hyperliquid_monitor_plus.monitoring import (
    PerformanceMonitor,
    ResourceMonitor,
    MetricsCollector,
    OptimizationEngine
)
```

### Performance Features

- **Real-time Metrics**: Latency, throughput, memory usage
- **Resource Monitoring**: CPU, memory, network utilization
- **Database Optimization**: Query performance and indexing
- **WebSocket Health**: Connection monitoring and auto-reconnect
- **API Rate Limiting**: Compliance and optimization

### Performance Monitoring Examples

```python
# Initialize performance monitoring
perf_monitor = PerformanceMonitor()
resource_monitor = ResourceMonitor()
metrics_collector = MetricsCollector()

# Start monitoring
perf_monitor.start_monitoring()
resource_monitor.start_monitoring()

# Get performance statistics
stats = perf_monitor.get_performance_stats()
print(f"⚡ Performance Statistics:")
print(f"   Average Latency: {stats.avg_latency_ms:.1f}ms")
print(f"   95th Percentile Latency: {stats.p95_latency_ms:.1f}ms")
print(f"   Throughput: {stats.trades_per_second:.1f} trades/sec")
print(f"   Error Rate: {stats.error_rate:.2%}")
print(f"   Uptime: {stats.uptime_hours:.1f} hours")

# Resource usage monitoring
resource_stats = resource_monitor.get_resource_stats()
print(f"💻 Resource Usage:")
print(f"   CPU Usage: {resource_stats.cpu_usage:.1f}%")
print(f"   Memory Usage: {resource_stats.memory_usage_mb:.1f} MB")
print(f"   Memory Peak: {resource_stats.memory_peak_mb:.1f} MB")
print(f"   Disk I/O: {resource_stats.disk_io_mb:.1f} MB/s")
print(f"   Network I/O: {resource_stats.network_io_mb:.1f} MB/s")

# Database performance
db_stats = perf_monitor.get_database_stats()
print(f"💾 Database Performance:")
print(f"   Query Response Time: {db_stats.avg_query_time_ms:.1f}ms")
print(f"   Connection Pool: {db_stats.active_connections}/{db_stats.max_connections}")
print(f"   Cache Hit Rate: {db_stats.cache_hit_rate:.1%}")
print(f"   Slow Queries: {db_stats.slow_query_count}")

# WebSocket connection health
ws_stats = perf_monitor.get_websocket_stats()
print(f"🔌 WebSocket Health:")
print(f"   Active Connections: {ws_stats.active_connections}")
print(f"   Reconnection Rate: {ws_stats.reconnection_rate:.2%}")
print(f"   Message Rate: {ws_stats.messages_per_second:.1f}/sec")
print(f"   Connection Quality: {ws_stats.connection_quality:.1%}")

# Performance optimization suggestions
optimization_engine = OptimizationEngine()
suggestions = optimization_engine.analyze_performance(stats)
print(f"🔧 Optimization Suggestions:")
for suggestion in suggestions:
    print(f"   • {suggestion.title}")
    print(f"     Impact: {suggestion.impact}")
    print(f"     Implementation: {suggestion.implementation}")
```

### Performance Optimization Configuration

```python
# Performance optimization settings
optimization_config = {
    "database": {
        "connection_pool_size": 20,
        "query_timeout_ms": 5000,
        "enable_wal_mode": True,
        "enable_cache": True,
        "cache_size_mb": 512
    },
    
    "websocket": {
        "max_reconnect_attempts": 5,
        "reconnect_delay_ms": 1000,
        "heartbeat_interval_ms": 30000,
        "message_buffer_size": 1000
    },
    
    "api": {
        "rate_limit_requests": 100,
        "rate_limit_window_ms": 60000,
        "timeout_ms": 10000,
        "retry_attempts": 3
    },
    
    "monitoring": {
        "metrics_collection_interval": 10000,  # 10 seconds
        "performance_logging": True,
        "alert_thresholds": {
            "latency_ms": 100,
            "error_rate": 0.01,
            "memory_usage_mb": 1024
        }
    }
}

# Apply optimizations
optimization_engine = OptimizationEngine(optimization_config)
await optimization_engine.apply_optimizations()
```

### Performance Benchmarking

```python
class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self, whale_bot):
        self.whale_bot = whale_bot
        
    async def run_comprehensive_benchmark(self):
        """Run full performance benchmark suite"""
        results = {}
        
        # Trade processing benchmark
        results["trade_processing"] = await self.benchmark_trade_processing()
        
        # Database operations benchmark
        results["database"] = await self.benchmark_database_operations()
        
        # WebSocket performance benchmark
        results["websocket"] = await self.benchmark_websocket_performance()
        
        # Memory usage benchmark
        results["memory"] = await self.benchmark_memory_usage()
        
        # Alert system benchmark
        results["alerts"] = await self.benchmark_alert_system()
        
        return results
        
    async def benchmark_trade_processing(self):
        """Benchmark trade processing performance"""
        # Generate test trades
        test_trades = self.generate_test_trades(count=1000)
        
        start_time = time.time()
        processed_count = 0
        
        for trade in test_trades:
            await self.whale_bot.process_trade(trade)
            processed_count += 1
            
        end_time = time.time()
        
        return {
            "total_trades": processed_count,
            "processing_time_seconds": end_time - start_time,
            "trades_per_second": processed_count / (end_time - start_time),
            "avg_latency_ms": ((end_time - start_time) / processed_count) * 1000
        }
        
    def generate_test_trades(self, count=100):
        """Generate test trade data"""
        import random
        
        test_trades = []
        for i in range(count):
            trade = {
                "coin": random.choice(["BTC", "ETH", "SOL"]),
                "side": random.choice(["BUY", "SELL"]),
                "size": random.uniform(0.1, 100),
                "price": random.uniform(20000, 50000),
                "address": f"0x{random.randint(1000000000000000000000000000000000000000, 9999999999999999999999999999999999999999):040x}",
                "timestamp": datetime.now()
            }
            test_trades.append(trade)
            
        return test_trades

# Run benchmarks
benchmark = PerformanceBenchmark(whale_bot)
results = await benchmark.run_comprehensive_benchmark()

print(f"🏁 Benchmark Results:")
for category, result in results.items():
    print(f"\n📊 {category.upper()}:")
    for metric, value in result.items():
        print(f"   {metric}: {value}")
```

---

## 📖 API Reference

### Core Classes

#### `HyperliquidMonitor`
Main monitoring class for tracking whale trades.

**Parameters:**
- `addresses` (List[str]): List of Ethereum addresses to monitor
- `db_path` (Optional[str]): Path to SQLite database for storage
- `callback` (Optional[TradeCallback]): Callback function for each trade
- `silent` (bool): Suppress callback notifications
- `alert_callback` (Optional[AlertCallback]): Callback for alerts
- `pnl_callback` (Optional[PnLCallback]): Callback for PnL events
- `track_pnl` (bool): Enable automatic PnL tracking

**Methods:**
- `start()`: Start monitoring
- `stop()`: Stop monitoring
- `cleanup()`: Clean up resources

#### `Database` / `TradeDatabase`
SQLite database for persistent trade storage.

**Methods:**
- `store_fill(fill, address)`: Store a trade fill
- `get_recent_trades(limit)`: Get recent trades
- `get_trades_by_coin(coin, limit)`: Filter by coin
- `get_trades_by_address(address, limit)`: Filter by address
- `get_trades_by_date_range(start, end)`: Filter by date
- `get_trade_statistics(coin)`: Get statistics
- `get_unique_coins()`: List all traded coins
- `get_unique_addresses()`: List all monitored addresses
- `close()`: Close database connection

#### `PnLManager`
Profit & Loss tracking and analysis.

**Methods:**
- `process_trade(trade)`: Process trade and update PnL
- `update_prices(prices)`: Update current prices
- `get_position(coin)`: Get position for coin
- `get_all_positions()`: Get all positions
- `get_open_positions()`: Get only open positions
- `get_coin_pnl(coin)`: Get PnL statistics for coin
- `get_portfolio_pnl()`: Get overall portfolio PnL
- `get_statistics()`: Get comprehensive statistics
- `get_summary_report()`: Generate text report
- `reset()`: Reset all tracking data

#### `PositionTracker`
Real-time position tracking from exchange.

**Methods:**
- `fetch_positions()`: Fetch current positions
- `get_position(coin)`: Get specific position
- `get_all_positions()`: Get all positions
- `get_open_positions()`: Get open positions only
- `get_account_state()`: Get account state
- `get_risky_positions(threshold)`: Get at-risk positions
- `get_statistics()`: Get position statistics
- `get_summary_report()`: Generate text report

#### `AlertManager`
Smart alert system with conditions.

**Methods:**
- `add_condition(condition)`: Add alert condition
- `remove_condition(name)`: Remove condition
- `update_condition(name, **kwargs)`: Update condition
- `enable_condition(name)`: Enable condition
- `disable_condition(name)`: Disable condition
- `get_condition(name)`: Get specific condition
- `list_conditions()`: List all conditions
- `check_trade(trade)`: Check trade against conditions
- `get_history(limit)`: Get alert history
- `get_statistics()`: Get alert statistics

#### `WhaleBotCore`
Advanced whale bot with intelligence.

**Methods:**
- `start_monitoring(addresses, db_path)`: Start monitoring
- `stop_monitoring()`: Stop monitoring
- `add_detection_callback(callback)`: Add whale detection callback
- `add_follow_callback(callback)`: Add follow recommendation callback
- `get_bot_statistics()`: Get comprehensive statistics
- `start_advanced_features()`: Start Phase 3 features (async)
- `run_ml_prediction(trade)`: Run ML prediction (async)
- `track_cross_chain_activity(trade)`: Track cross-chain (async)
- `optimize_portfolio()`: Run portfolio optimization (async)

### Type Classes

#### `Trade`
Basic trade data structure.

**Attributes:**
- `timestamp`: Trade timestamp
- `address`: Wallet address
- `coin`: Coin symbol
- `side`: "BUY" or "SELL"
- `size`: Trade size
- `price`: Trade price
- `trade_type`: "FILL"
- `tx_hash`: Transaction hash
- `fee`: Trade fee
- `closed_pnl`: Closed PnL
- `leverage`: Position leverage

#### `EnhancedTrade`
Extended trade with whale intelligence.

**Additional Attributes:**
- `whale_tier`: WhaleTier enum (MINNOW to GIANT_WHALE)
- `urgency`: Trade urgency level
- `size_usd`: USD value
- `whale_direction`: "LONG", "SHORT", or "UNKNOWN"
- `market_impact`: Predicted price impact
- `slippage`: Expected slippage
- `follow_opportunity`: Follow recommendation

#### `Position`
Position tracking (PnL system).

**Attributes:**
- `coin`: Coin symbol
- `side`: "LONG", "SHORT", or "NEUTRAL"
- `size`: Position size
- `avg_entry_price`: Average entry price
- `realized_pnl`: Realized profit/loss

#### `LivePosition`
Live position from exchange.

**Attributes:**
- `coin`: Coin symbol
- `side`: "LONG", "SHORT", or "NEUTRAL"
- `size`: Position size
- `entry_price`: Entry price
- `mark_price`: Current mark price
- `liquidation_price`: Liquidation price
- `unrealized_pnl`: Unrealized PnL
- `margin_used`: Margin used
- `leverage`: Position leverage
- `pnl_percentage`: PnL percentage
- `distance_to_liquidation`: Distance to liquidation (%)

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Hyperliquid API (optional - uses mainnet by default)
HYPERLIQUID_USE_TESTNET=false

# Discord Notifications (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Database (optional)
DATABASE_PATH=whale_trades.db

# Whale Thresholds (optional)
MIN_WHALE_THRESHOLD=100000
MEGA_WHALE_THRESHOLD=1000000
GIANT_WHALE_THRESHOLD=5000000

# ML & Advanced Features (optional)
ENABLE_ML_MODELS=true
ENABLE_CROSS_CHAIN=true
ENABLE_DASHBOARD=true
DASHBOARD_PORT=8080
```

### Configuration Files

Create `config.yaml`:

```yaml
whale_bot:
  # Thresholds
  min_whale_threshold: 100000
  mega_whale_threshold: 1000000
  giant_whale_threshold: 5000000
  
  # Performance
  max_response_time_ms: 100
  max_whales: 10
  
  # Trading
  follow_ratio: 0.1
  max_position_risk: 0.05
  max_market_impact_threshold: 0.01
  max_slippage_threshold: 0.005
  
  # Alerts
  enable_real_time_alerts: true
  alert_cooldown_seconds: 30
  
  # Features
  dashboard: false
  track_pnl: true
  
phase3:
  ml_model_enabled: true
  cross_chain_enabled: true
  dashboard_enabled: true
  dashboard_port: 8080
  portfolio_optimization_enabled: true
  
notifications:
  discord:
    enabled: true
    webhook_url: ${DISCORD_WEBHOOK_URL}
  
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hyperliquid_monitor_plus --cov-report=html

# Run specific test module
pytest tests/test_monitor.py

# Run with verbose output
pytest -v

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

---

## 📊 Examples

The library includes comprehensive examples in the `examples/` directory:

### Basic Examples
- `examples/monitoring/basic_monitoring.py` - Basic whale monitoring
- `examples/database/data_analysis.py` - Database queries and analysis
- `examples/notifications/alert_system.py` - Alert configuration

### Advanced Examples
- `examples/core/whale_bot_example.py` - Full whale bot setup
- `examples/analysis/market_analysis.py` - Market impact analysis
- `examples/tracking/advanced_tracking.py` - Advanced position tracking
- `examples/strategy/strategy_analyzer.py` - Strategy analysis
- `examples/intelligence/intelligence_demo.py` - ML and intelligence
- `examples/integration/dashboard_integration.py` - Dashboard setup
- `examples/configuration/advanced_config.py` - Advanced configuration

Run any example:

```bash
python -m hyperliquid_monitor_plus.examples.monitoring.basic_monitoring
python -m hyperliquid_monitor_plus.examples.core.whale_bot_example
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Pezhman5252/hyperliquid_monitor_plus.git
cd hyperliquid_monitor_plus

# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black hyperliquid_monitor_plus/

# Lint code
flake8 hyperliquid_monitor_plus/

# Type checking
mypy hyperliquid_monitor_plus/
```

### Guidelines

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass
5. Use type hints

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on top of the official [Hyperliquid Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk)
- Inspired by the amazing Hyperliquid community
- Special thanks to all contributors

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Pezhman5252/hyperliquid_monitor_plus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Pezhman5252/hyperliquid_monitor_plus/discussions)
- **Email**: Prtsianboy.1991g@gmail.com

---

## 💝 Support the Developer

### 🚀 Why Your Support Matters

Developing and maintaining high-quality, production-ready libraries requires significant time, effort, and dedication. Your support helps me:

- **🎯 Continue Innovation**: Invest in new features and cutting-edge developments
- **🔧 Maintain Quality**: Ensure the library stays reliable, secure, and up-to-date
- **🌟 Build More Tools**: Create additional powerful trading and blockchain analysis tools
- **📚 Improve Documentation**: Enhance guides, examples, and learning resources
- **⚡ Provide Support**: Offer better customer service and community engagement

### 💰 Crypto Donations

If you find this library valuable for your trading or analysis, consider supporting its continued development:

**🔗 Wallet Address:**
```
0xB05675CE390c895133dE8Aa1A873484f1FA1Df2C
```

**Supported Networks:**
- 🟦 **Ethereum (ETH)** - Full support
- ⚡ **Arbitrum (ARB)** - Low fees, fast transactions  
- 🔵 **Polygon (MATIC)** - Ultra-low fees
- ⚡ **Optimism (OP)** - Fast Layer 2 transactions
- 🌟 **Base Network** - Coinbase Layer 2

**💡 Quick Donation Guide:**

**For Desktop Users:**
1. Open your crypto wallet (MetaMask, Rabby, etc.)
2. Select the appropriate network
3. Copy the address: `0xB05675CE390c895133dE8Aa1A873484f1FA1Df2C`
4. Send any amount you feel appropriate

**For Mobile Users:**
1. Use Rabby Wallet, MetaMask Mobile, or similar
2. Tap "Send" and paste the address
3. Choose your preferred network for optimal fees

**🎁 Any Contribution is Appreciated:**
- Even small donations help cover hosting and development costs
- Your support directly fuels innovation in DeFi analytics
- Every contribution enables more features and improvements

**🤝 Community Impact:**
Your donations don't just support development—they help build a thriving ecosystem of tools that benefit the entire DeFi community. Together, we're making blockchain analytics more accessible and powerful!

---

## 🗺️ Roadmap

### Completed ✅
- [x] Core monitoring system
- [x] Multi-address support
- [x] Database storage
- [x] PnL tracking
- [x] Position tracking
- [x] Alert system
- [x] Whale bot intelligence
- [x] ML predictions
- [x] Cross-chain tracking
- [x] Real-time dashboard
- [x] Portfolio optimization

### Planned 🔜
- [ ] Advanced charting and visualization
- [ ] Automated trading execution
- [ ] Risk management system
- [ ] Backtesting framework
- [ ] Mobile app integration
- [ ] More ML models
- [ ] Additional DEX support
- [ ] Cloud deployment guides

---

## ⚡ Performance

- **Real-time Processing**: < 100ms latency for trade detection
- **Concurrent Support**: Thread-safe for multiple addresses
- **Database**: Optimized SQLite with indexes
- **Memory Efficient**: Circular buffers for history
- **Scalable**: Handles 1000+ trades/minute

---

## 🔐 Security

- No private keys stored or required
- Read-only API access
- Secure WebSocket connections
- Safe database operations
- Input validation and sanitization

---

## 📈 Version History

### v3.0.0 (Latest)
- Added Phase 3: ML & Advanced Features
- Cross-chain whale tracking
- Portfolio optimization
- Real-time dashboard
- Improved performance and stability

### v2.0.0
- Major refactoring for production
- Added PnL tracking system
- Position tracking from exchange
- Strategy analyzer
- Report generation
- Notification integrations

### v1.0.0
- Initial release
- Basic whale monitoring
- Database storage
- Alert system

---

**Made with ❤️ by Pezhman Hajipour**

*Star ⭐ this repository if you find it helpful!*
