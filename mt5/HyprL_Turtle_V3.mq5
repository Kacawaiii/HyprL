//+------------------------------------------------------------------+
//|                                         HyprL_Turtle_V3.mq5      |
//|                          Turtle Trading System - Production V3    |
//|                             Based on Richard Dennis Rules         |
//|                       PRODUCTION READY - ALL CRITICAL FIXES       |
//+------------------------------------------------------------------+
#property copyright "HyprL Trading Systems"
#property link      "https://hyprlcore.com"
#property version   "3.00"
#property description "Turtle Trading V3 - Production Ready"
#property description "CRITICAL FIXES: Dynamic Fill Mode, Circuit Breaker Reset, Spread Filter"
#property description "Session Filter, State Persistence, Max DD Kill Switch, OnTrade Fix"
#property description "Features: Donchian Breakout, ATR Sizing, ADX Filter, Skip Rule, HTF Confirmation"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+

//--- Strategy Parameters
input group "=== STRATEGY SETTINGS ==="
input int      InpEntryPeriod     = 20;        // Entry Breakout Period (S1)
input int      InpExitPeriod      = 10;        // Exit Breakout Period
input int      InpFailsafePeriod  = 55;        // Failsafe Breakout Period (S2)
input int      InpATRPeriod       = 20;        // ATR Period (Wilder)

//--- Risk Management
input group "=== RISK MANAGEMENT ==="
input double   InpRiskPercent     = 1.0;       // Risk % Per Trade
input double   InpATRStopMult     = 2.0;       // Stop Loss (x ATR)
input double   InpATRTPMult       = 4.0;       // Take Profit (x ATR) - 0=disabled
input double   InpMaxTotalRisk    = 6.0;       // Max Total Open Risk %

//--- ADX Filter
input group "=== ADX TREND FILTER ==="
input bool     InpUseADXFilter    = true;      // Enable ADX Filter
input int      InpADXPeriod       = 14;        // ADX Period
input int      InpADXThreshold    = 25;        // ADX Minimum for Trade
input int      InpADXReduceLevel  = 20;        // ADX Level for Reduced Size

//--- Skip After Winner Rule
input group "=== SKIP AFTER WINNER ==="
input bool     InpUseSkipRule     = true;      // Enable Skip After Winner
input bool     InpUseFailsafe     = true;      // Enable 55-Day Failsafe

//--- Higher Timeframe Confirmation
input group "=== HTF CONFIRMATION ==="
input bool     InpUseHTFFilter    = true;      // Enable HTF Confirmation
input ENUM_TIMEFRAMES InpHTFTimeframe = PERIOD_H4; // HTF Timeframe
input int      InpHTFEMAPeriod    = 50;        // HTF EMA Period

//--- Spread Filter (NEW V3)
input group "=== SPREAD FILTER ==="
input bool     InpUseSpreadFilter = true;      // Enable Spread Filter
input double   InpMaxSpreadATR    = 0.3;       // Max Spread as % of ATR (0.3 = 30%)

//--- Session Filter (NEW V3)
input group "=== SESSION FILTER ==="
input bool     InpUseSessionFilter = true;     // Enable Session Filter
input string   InpSessionStart     = "08:00";  // Session Start (GMT)
input string   InpSessionEnd       = "20:00";  // Session End (GMT)
input int      InpGMTOffset        = 0;        // Broker GMT Offset

//--- Trailing Stop
input group "=== TRAILING STOP ==="
input bool     InpUseTrailing     = true;      // Enable Trailing Stop
input double   InpTrailActivation = 1.5;       // Activation (x ATR profit)
input double   InpTrailDistance   = 1.0;       // Trail Distance (x ATR)

//--- Pyramiding
input group "=== PYRAMIDING ==="
input bool     InpUsePyramiding   = true;      // Enable Pyramiding
input double   InpPyramidStep     = 0.5;       // Add Position Every (x ATR)
input int      InpMaxUnits        = 4;         // Max Units Per Symbol

//--- Circuit Breakers
input group "=== CIRCUIT BREAKERS ==="
input bool     InpUseCircuitBreaker = true;    // Enable Circuit Breakers
input double   InpDailyLossLimit  = 3.0;       // Daily Loss Limit %
input double   InpWeeklyLossLimit = 6.0;       // Weekly Loss Limit %
input int      InpConsecLossLimit = 3;         // Consecutive Losses Pause

//--- Max Drawdown Kill Switch (NEW V3)
input group "=== MAX DRAWDOWN KILL SWITCH ==="
input bool     InpUseMaxDD        = true;      // Enable Max DD Kill Switch
input double   InpMaxDrawdownPct  = 10.0;      // Max Drawdown % (permanent stop)

//--- General
input group "=== GENERAL ==="
input ulong    InpMagicNumber     = 2024001;   // Magic Number
input string   InpTradeComment    = "TurtleV3"; // Trade Comment

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
CTrade         trade;
CPositionInfo  posInfo;
CAccountInfo   accInfo;

//--- Indicator Handles (initialized to INVALID_HANDLE to prevent accidental release)
int handleATR     = INVALID_HANDLE;
int handleADX     = INVALID_HANDLE;
int handleHTF_EMA = INVALID_HANDLE;

//--- State Variables
bool     lastTradeWasWinner = false;    // For Skip After Winner rule
bool     skipNextS1Signal   = false;    // Skip flag
datetime lastTradeCloseTime = 0;        // Last trade close time
double   lastTradeProfit    = 0;        // Last trade profit
int      currentUnits       = 0;        // Current pyramid units
double   lastEntryPrice     = 0;        // Last entry price for pyramiding
int      consecutiveLosses  = 0;        // For circuit breaker
double   dailyStartEquity   = 0;        // For daily loss tracking
double   weeklyStartEquity  = 0;        // For weekly loss tracking
datetime lastDayChecked     = 0;        // Day tracking
datetime lastWeekChecked    = 0;        // Week tracking
bool     dailyPaused        = false;    // Daily circuit breaker active
bool     weeklyReduced      = false;    // Weekly size reduction active

//--- NEW V3: Max DD Kill Switch
double   peakEquity         = 0;        // High-water mark
bool     maxDDKilled        = false;    // Permanent kill flag

//--- Donchian Channel Values
double   highestHigh20, lowestLow20;    // S1 Entry
double   highestHigh10, lowestLow10;    // S1 Exit
double   highestHigh55, lowestLow55;    // S2 Failsafe

//--- NEW V3: State persistence prefix
string globalVarPrefix;

//+------------------------------------------------------------------+
//| NEW V3: Dynamic Order Filling Mode Detection                     |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING DetectFillingMode()
{
    long filling = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
    if((filling & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
    if((filling & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
    return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| NEW V3: Save State to GlobalVariables                            |
//+------------------------------------------------------------------+
void SaveState()
{
    GlobalVariableSet(globalVarPrefix + "consLoss", consecutiveLosses);
    GlobalVariableSet(globalVarPrefix + "skipS1", skipNextS1Signal ? 1.0 : 0.0);
    GlobalVariableSet(globalVarPrefix + "dailyEq", dailyStartEquity);
    GlobalVariableSet(globalVarPrefix + "weeklyEq", weeklyStartEquity);
    GlobalVariableSet(globalVarPrefix + "peakEq", peakEquity);
    GlobalVariableSet(globalVarPrefix + "maxDDKill", maxDDKilled ? 1.0 : 0.0);
    GlobalVariableSet(globalVarPrefix + "units", currentUnits);
    GlobalVariableSet(globalVarPrefix + "lastEntry", lastEntryPrice);
    GlobalVariableSet(globalVarPrefix + "lastDay", lastDayChecked);
    GlobalVariableSet(globalVarPrefix + "lastWeek", lastWeekChecked);
}

//+------------------------------------------------------------------+
//| NEW V3: Load State from GlobalVariables                          |
//+------------------------------------------------------------------+
void LoadState()
{
    if(GlobalVariableCheck(globalVarPrefix + "consLoss"))
    {
        consecutiveLosses = (int)GlobalVariableGet(globalVarPrefix + "consLoss");
        skipNextS1Signal = GlobalVariableGet(globalVarPrefix + "skipS1") > 0.5;
        dailyStartEquity = GlobalVariableGet(globalVarPrefix + "dailyEq");
        weeklyStartEquity = GlobalVariableGet(globalVarPrefix + "weeklyEq");
        peakEquity = GlobalVariableGet(globalVarPrefix + "peakEq");
        maxDDKilled = GlobalVariableGet(globalVarPrefix + "maxDDKill") > 0.5;
        currentUnits = (int)GlobalVariableGet(globalVarPrefix + "units");
        lastEntryPrice = GlobalVariableGet(globalVarPrefix + "lastEntry");
        lastDayChecked = (datetime)GlobalVariableGet(globalVarPrefix + "lastDay");
        lastWeekChecked = (datetime)GlobalVariableGet(globalVarPrefix + "lastWeek");

        Print("✅ State restored from GlobalVariables:");
        Print("   Consecutive Losses: ", consecutiveLosses);
        Print("   Skip Next S1: ", skipNextS1Signal);
        Print("   Peak Equity: ", DoubleToString(peakEquity, 2));
        Print("   Max DD Killed: ", maxDDKilled);
        Print("   Current Units: ", currentUnits);
    }
    else
    {
        Print("📝 No saved state found, starting fresh");
        peakEquity = accInfo.Equity();
        dailyStartEquity = accInfo.Equity();
        weeklyStartEquity = accInfo.Equity();
        SaveState();
    }
}

//+------------------------------------------------------------------+
//| NEW V3: Check if spread is acceptable                            |
//+------------------------------------------------------------------+
bool IsSpreadAcceptable(double atr)
{
    if(!InpUseSpreadFilter) return true;

    double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double maxSpread = atr * InpMaxSpreadATR;

    if(spread > maxSpread)
    {
        Print("⏸️ Spread too wide: ", DoubleToString(spread, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              " > ", DoubleToString(maxSpread, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              " (", DoubleToString(InpMaxSpreadATR * 100, 0), "% of ATR)");
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| NEW V3: Check if within trading session                          |
//+------------------------------------------------------------------+
bool IsWithinSession()
{
    if(!InpUseSessionFilter) return true;

    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);

    // Adjust for broker GMT offset
    dt.hour = (dt.hour + InpGMTOffset + 24) % 24;

    // Parse session start/end
    string parts[];
    int startHour = 0, startMin = 0, endHour = 23, endMin = 59;

    if(StringSplit(InpSessionStart, ':', parts) == 2)
    {
        startHour = (int)StringToInteger(parts[0]);
        startMin = (int)StringToInteger(parts[1]);
    }

    if(StringSplit(InpSessionEnd, ':', parts) == 2)
    {
        endHour = (int)StringToInteger(parts[0]);
        endMin = (int)StringToInteger(parts[1]);
    }

    int currentMinutes = dt.hour * 60 + dt.min;
    int startMinutes = startHour * 60 + startMin;
    int endMinutes = endHour * 60 + endMin;

    // Handle midnight crossover
    if(endMinutes < startMinutes)
    {
        // Session crosses midnight (e.g., 20:00 - 08:00)
        return (currentMinutes >= startMinutes || currentMinutes <= endMinutes);
    }
    else
    {
        // Normal session (e.g., 08:00 - 20:00)
        return (currentMinutes >= startMinutes && currentMinutes <= endMinutes);
    }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Initialize global variable prefix
    globalVarPrefix = "TV3_" + _Symbol + "_" + IntegerToString(InpMagicNumber) + "_";

    //--- Initialize trade object
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(10);

    //--- FIX #1: Dynamic filling mode detection (replaces hardcoded IOC)
    ENUM_ORDER_TYPE_FILLING fillingMode = DetectFillingMode();
    trade.SetTypeFilling(fillingMode);
    Print("📋 Order filling mode: ", EnumToString(fillingMode));

    //--- Create ATR indicator
    handleATR = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
    if(handleATR == INVALID_HANDLE)
    {
        Print("Error creating ATR indicator");
        return INIT_FAILED;
    }

    //--- Create ADX indicator if enabled
    if(InpUseADXFilter)
    {
        handleADX = iADX(_Symbol, PERIOD_CURRENT, InpADXPeriod);
        if(handleADX == INVALID_HANDLE)
        {
            Print("Error creating ADX indicator");
            return INIT_FAILED;
        }
    }

    //--- Create HTF EMA if enabled
    if(InpUseHTFFilter)
    {
        handleHTF_EMA = iMA(_Symbol, InpHTFTimeframe, InpHTFEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
        if(handleHTF_EMA == INVALID_HANDLE)
        {
            Print("Error creating HTF EMA indicator");
            return INIT_FAILED;
        }
    }

    //--- FIX #5: Load state from GlobalVariables
    LoadState();

    Print("═══════════════════════════════════");
    Print("   🐢 TURTLE BOT V3 INITIALIZED 🐢");
    Print("═══════════════════════════════════");
    Print("Symbol: ", _Symbol);
    Print("Magic Number: ", InpMagicNumber);
    Print("");
    Print("FEATURES:");
    Print("  ADX Filter: ", InpUseADXFilter ? "ON (>=" + IntegerToString(InpADXThreshold) + ")" : "OFF");
    Print("  Skip After Winner: ", InpUseSkipRule ? "ON" : "OFF");
    Print("  55-Day Failsafe: ", InpUseFailsafe ? "ON" : "OFF");
    Print("  HTF Confirmation: ", InpUseHTFFilter ? "ON (" + EnumToString(InpHTFTimeframe) + ")" : "OFF");
    Print("  Spread Filter: ", InpUseSpreadFilter ? "ON (max " + DoubleToString(InpMaxSpreadATR * 100, 0) + "% ATR)" : "OFF");
    Print("  Session Filter: ", InpUseSessionFilter ? "ON (" + InpSessionStart + "-" + InpSessionEnd + " GMT)" : "OFF");
    Print("  Circuit Breakers: ", InpUseCircuitBreaker ? "ON" : "OFF");
    Print("  Max DD Kill Switch: ", InpUseMaxDD ? "ON (" + DoubleToString(InpMaxDrawdownPct, 1) + "%)" : "OFF");
    Print("  Pyramiding: ", InpUsePyramiding ? "ON (max " + IntegerToString(InpMaxUnits) + ")" : "OFF");
    Print("═══════════════════════════════════");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(handleATR != INVALID_HANDLE) IndicatorRelease(handleATR);
    if(handleADX != INVALID_HANDLE) IndicatorRelease(handleADX);
    if(handleHTF_EMA != INVALID_HANDLE) IndicatorRelease(handleHTF_EMA);

    SaveState();
    Print("TurtleBot V3 Deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Check for new bar
    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;

    //--- NEW V3: Check session filter first
    if(!IsWithinSession())
    {
        Comment("⏸️ Outside trading session - waiting for " + InpSessionStart + "-" + InpSessionEnd + " GMT");
        return;
    }

    //--- Update circuit breakers (includes FIX #2 and FIX #6)
    UpdateCircuitBreakers();

    //--- Check if trading is paused
    if(dailyPaused)
    {
        if(maxDDKilled)
            Comment("💀 MAX DRAWDOWN KILL SWITCH ACTIVE - EA PERMANENTLY STOPPED");
        else
            Comment("⛔ CIRCUIT BREAKER ACTIVE - Trading paused");
        return;
    }

    //--- Update Donchian Channels
    CalculateDonchianChannels();

    //--- Get current ATR
    double atr = GetATR();
    if(atr <= 0) return;

    //--- NEW V3: Check spread filter
    if(!IsSpreadAcceptable(atr))
    {
        ManagePositions(atr);
        UpdateComment(atr, 0, 0);
        return;
    }

    //--- Get current ADX if enabled
    double adx = 100; // Default to always trade if disabled
    if(InpUseADXFilter)
    {
        adx = GetADX();
        if(adx < InpADXReduceLevel)
        {
            Comment("⏸️ ADX = ", DoubleToString(adx, 1), " < ", InpADXReduceLevel, " - No trend, waiting...");
            ManagePositions(atr);
            return;
        }
    }

    //--- Get HTF trend direction
    int htfTrend = 0; // 0 = neutral, 1 = up, -1 = down
    if(InpUseHTFFilter)
    {
        htfTrend = GetHTFTrend();
    }

    //--- Manage existing positions (trailing, exits)
    ManagePositions(atr);

    //--- Check for entry signals
    CheckEntrySignals(atr, adx, htfTrend);

    //--- Update comment
    UpdateComment(atr, adx, htfTrend);
}

//+------------------------------------------------------------------+
//| Calculate Donchian Channels                                       |
//+------------------------------------------------------------------+
void CalculateDonchianChannels()
{
    //--- S1 Entry (20-day) - start=2 to exclude comparison bar (bar 1)
    highestHigh20 = iHigh(_Symbol, PERIOD_CURRENT, iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, InpEntryPeriod, 2));
    lowestLow20 = iLow(_Symbol, PERIOD_CURRENT, iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, InpEntryPeriod, 2));

    //--- S1 Exit (10-day) - start=2 to exclude comparison bar (bar 1)
    highestHigh10 = iHigh(_Symbol, PERIOD_CURRENT, iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, InpExitPeriod, 2));
    lowestLow10 = iLow(_Symbol, PERIOD_CURRENT, iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, InpExitPeriod, 2));

    //--- S2 Failsafe (55-day) - start=2 to exclude comparison bar (bar 1)
    if(InpUseFailsafe)
    {
        highestHigh55 = iHigh(_Symbol, PERIOD_CURRENT, iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, InpFailsafePeriod, 2));
        lowestLow55 = iLow(_Symbol, PERIOD_CURRENT, iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, InpFailsafePeriod, 2));
    }
}

//+------------------------------------------------------------------+
//| Get ATR value                                                     |
//+------------------------------------------------------------------+
double GetATR()
{
    double atrBuffer[];
    ArraySetAsSeries(atrBuffer, true);
    if(CopyBuffer(handleATR, 0, 0, 1, atrBuffer) <= 0) return 0;
    return atrBuffer[0];
}

//+------------------------------------------------------------------+
//| Get ADX value                                                     |
//+------------------------------------------------------------------+
double GetADX()
{
    double adxBuffer[];
    ArraySetAsSeries(adxBuffer, true);
    if(CopyBuffer(handleADX, 0, 0, 1, adxBuffer) <= 0) return 0;
    return adxBuffer[0];
}

//+------------------------------------------------------------------+
//| Get Higher Timeframe Trend                                        |
//+------------------------------------------------------------------+
int GetHTFTrend()
{
    double emaBuffer[];
    ArraySetAsSeries(emaBuffer, true);
    if(CopyBuffer(handleHTF_EMA, 0, 0, 1, emaBuffer) <= 0) return 0;

    double htfClose = iClose(_Symbol, InpHTFTimeframe, 0);

    if(htfClose > emaBuffer[0]) return 1;   // Uptrend
    if(htfClose < emaBuffer[0]) return -1;  // Downtrend
    return 0; // Neutral
}

//+------------------------------------------------------------------+
//| Check Entry Signals                                               |
//+------------------------------------------------------------------+
void CheckEntrySignals(double atr, double adx, int htfTrend)
{
    //--- Get current price
    double close = iClose(_Symbol, PERIOD_CURRENT, 1);
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    //--- Check if we have open positions
    int posCount = CountPositions();

    //--- Calculate size multiplier based on ADX and weekly reduction
    double sizeMultiplier = 1.0;
    if(InpUseADXFilter && adx < InpADXThreshold && adx >= InpADXReduceLevel)
    {
        sizeMultiplier = 0.5; // Reduced size in weak trend
    }
    if(weeklyReduced)
    {
        sizeMultiplier *= 0.5; // Further reduce if weekly limit hit
    }

    //--- LONG SIGNAL
    bool longBreakoutS1 = close > highestHigh20;
    bool longBreakoutS2 = InpUseFailsafe && close > highestHigh55;

    if(longBreakoutS1 || longBreakoutS2)
    {
        //--- Check HTF confirmation
        if(InpUseHTFFilter && htfTrend < 0)
        {
            // HTF is down, skip long signal
            return;
        }

        //--- Apply Skip After Winner rule for S1
        if(longBreakoutS1 && !longBreakoutS2)
        {
            if(InpUseSkipRule && skipNextS1Signal)
            {
                Print("SKIP: S1 Long signal skipped (last trade was winner)");
                return;
            }
        }

        //--- Failsafe S2 overrides skip rule
        if(longBreakoutS2 && skipNextS1Signal)
        {
            Print("FAILSAFE: Taking S2 (55-day) breakout despite skip rule");
            skipNextS1Signal = false; // Reset skip
            SaveState();
        }

        //--- Check if we can add position (pyramiding)
        if(posCount == 0)
        {
            // New position
            OpenPosition(ORDER_TYPE_BUY, atr, sizeMultiplier, "S1_Long");
            currentUnits = 1;
            lastEntryPrice = currentPrice;
            SaveState();
        }
        else if(InpUsePyramiding && posCount < InpMaxUnits)
        {
            // Pyramid: add if price moved enough
            if(currentPrice >= lastEntryPrice + (InpPyramidStep * atr))
            {
                OpenPosition(ORDER_TYPE_BUY, atr, sizeMultiplier, "Pyramid_Long_" + IntegerToString(posCount + 1));
                currentUnits++;
                lastEntryPrice = currentPrice;
                SaveState();
            }
        }
    }

    //--- SHORT SIGNAL
    bool shortBreakoutS1 = close < lowestLow20;
    bool shortBreakoutS2 = InpUseFailsafe && close < lowestLow55;

    if(shortBreakoutS1 || shortBreakoutS2)
    {
        //--- Check HTF confirmation
        if(InpUseHTFFilter && htfTrend > 0)
        {
            // HTF is up, skip short signal
            return;
        }

        //--- Apply Skip After Winner rule for S1
        if(shortBreakoutS1 && !shortBreakoutS2)
        {
            if(InpUseSkipRule && skipNextS1Signal)
            {
                Print("SKIP: S1 Short signal skipped (last trade was winner)");
                return;
            }
        }

        //--- Failsafe S2 overrides skip rule
        if(shortBreakoutS2 && skipNextS1Signal)
        {
            Print("FAILSAFE: Taking S2 (55-day) breakout despite skip rule");
            skipNextS1Signal = false;
            SaveState();
        }

        //--- Check if we can add position
        if(posCount == 0)
        {
            OpenPosition(ORDER_TYPE_SELL, atr, sizeMultiplier, "S1_Short");
            currentUnits = 1;
            lastEntryPrice = currentPrice;
            SaveState();
        }
        else if(InpUsePyramiding && posCount < InpMaxUnits)
        {
            if(currentPrice <= lastEntryPrice - (InpPyramidStep * atr))
            {
                OpenPosition(ORDER_TYPE_SELL, atr, sizeMultiplier, "Pyramid_Short_" + IntegerToString(posCount + 1));
                currentUnits++;
                lastEntryPrice = currentPrice;
                SaveState();
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Open Position                                                     |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE orderType, double atr, double sizeMultiplier, string comment)
{
    //--- Check max total risk
    double currentRisk = CalculateCurrentRisk();
    if(currentRisk >= InpMaxTotalRisk)
    {
        Print("MAX RISK: Current open risk ", DoubleToString(currentRisk, 2), "% >= ", InpMaxTotalRisk, "%");
        return;
    }

    //--- Calculate position size
    double stopDistance = atr * InpATRStopMult;
    double lotSize = CalculateLotSize(stopDistance, sizeMultiplier);

    if(lotSize <= 0)
    {
        Print("Error: Invalid lot size");
        return;
    }

    //--- Get current prices
    double price, sl, tp;

    if(orderType == ORDER_TYPE_BUY)
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        sl = price - stopDistance;
        tp = (InpATRTPMult > 0) ? price + (atr * InpATRTPMult) : 0;
    }
    else
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        sl = price + stopDistance;
        tp = (InpATRTPMult > 0) ? price - (atr * InpATRTPMult) : 0;
    }

    //--- Normalize prices
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    sl = NormalizeDouble(sl, digits);
    tp = NormalizeDouble(tp, digits);

    //--- Execute trade
    string fullComment = InpTradeComment + "_" + comment;

    if(trade.PositionOpen(_Symbol, orderType, lotSize, price, sl, tp, fullComment))
    {
        Print("✅ ", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), " ", lotSize, " lots @ ", price);
        Print("   SL: ", sl, " (", DoubleToString(stopDistance/_Point, 0), " pts)");
        Print("   TP: ", tp);
        Print("   ATR: ", DoubleToString(atr, digits));
    }
    else
    {
        Print("❌ Order failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
    }
}

//+------------------------------------------------------------------+
//| Calculate Lot Size                                                |
//+------------------------------------------------------------------+
double CalculateLotSize(double stopDistance, double sizeMultiplier)
{
    double accountEquity = accInfo.Equity();
    double riskAmount = accountEquity * (InpRiskPercent / 100.0) * sizeMultiplier;

    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    if(tickValue <= 0 || tickSize <= 0) return 0;

    double stopTicks = stopDistance / tickSize;
    double lotSize = riskAmount / (stopTicks * tickValue);

    //--- Normalize lot size
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

    return lotSize;
}

//+------------------------------------------------------------------+
//| Manage Open Positions                                             |
//+------------------------------------------------------------------+
void ManagePositions(double atr)
{
    double close = iClose(_Symbol, PERIOD_CURRENT, 1);

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!posInfo.SelectByIndex(i)) continue;
        if(posInfo.Symbol() != _Symbol) continue;
        if(posInfo.Magic() != InpMagicNumber) continue;

        double openPrice = posInfo.PriceOpen();
        double currentSL = posInfo.StopLoss();
        double profit = posInfo.Profit();

        //--- DONCHIAN EXIT (Original Turtle Rule)
        if(posInfo.PositionType() == POSITION_TYPE_BUY)
        {
            // Exit long if price drops below 10-day low
            if(close < lowestLow10)
            {
                ClosePosition(posInfo.Ticket(), profit);
                continue;
            }

            //--- TRAILING STOP
            if(InpUseTrailing)
            {
                double profitATR = (SymbolInfoDouble(_Symbol, SYMBOL_BID) - openPrice) / atr;
                if(profitATR >= InpTrailActivation)
                {
                    double newSL = SymbolInfoDouble(_Symbol, SYMBOL_BID) - (atr * InpTrailDistance);
                    newSL = NormalizeDouble(newSL, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));

                    if(newSL > currentSL)
                    {
                        trade.PositionModify(posInfo.Ticket(), newSL, posInfo.TakeProfit());
                    }
                }
            }
        }
        else // SELL
        {
            // Exit short if price rises above 10-day high
            if(close > highestHigh10)
            {
                ClosePosition(posInfo.Ticket(), profit);
                continue;
            }

            //--- TRAILING STOP
            if(InpUseTrailing)
            {
                double profitATR = (openPrice - SymbolInfoDouble(_Symbol, SYMBOL_ASK)) / atr;
                if(profitATR >= InpTrailActivation)
                {
                    double newSL = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + (atr * InpTrailDistance);
                    newSL = NormalizeDouble(newSL, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));

                    if(newSL < currentSL || currentSL == 0)
                    {
                        trade.PositionModify(posInfo.Ticket(), newSL, posInfo.TakeProfit());
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Close Position and Update Skip Rule                               |
//+------------------------------------------------------------------+
void ClosePosition(ulong ticket, double profit)
{
    if(trade.PositionClose(ticket))
    {
        Print("📤 Position closed. Profit: ", DoubleToString(profit, 2));
        // NOTE: Skip rule and consecutiveLosses are handled ONLY in OnTrade()
        // to avoid double-counting (OnTrade fires for this same close event)

        //--- Reset pyramid counter if all positions closed
        if(CountPositions() == 0)
        {
            currentUnits = 0;
            lastEntryPrice = 0;
        }

        lastTradeProfit = profit;
        SaveState();
    }
}

//+------------------------------------------------------------------+
//| Count Open Positions                                              |
//+------------------------------------------------------------------+
int CountPositions()
{
    int count = 0;
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(posInfo.SelectByIndex(i))
        {
            if(posInfo.Symbol() == _Symbol && posInfo.Magic() == InpMagicNumber)
                count++;
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Calculate Current Open Risk                                       |
//+------------------------------------------------------------------+
double CalculateCurrentRisk()
{
    double totalRisk = 0;
    double equity = accInfo.Equity();

    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(posInfo.SelectByIndex(i))
        {
            if(posInfo.Symbol() == _Symbol && posInfo.Magic() == InpMagicNumber)
            {
                double openPrice = posInfo.PriceOpen();
                double sl = posInfo.StopLoss();
                double lots = posInfo.Volume();

                if(sl > 0)
                {
                    double riskPoints = MathAbs(openPrice - sl);
                    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
                    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
                    double riskAmount = (riskPoints / tickSize) * tickValue * lots;
                    totalRisk += (riskAmount / equity) * 100;
                }
            }
        }
    }

    return totalRisk;
}

//+------------------------------------------------------------------+
//| FIX #2 & #6: Update Circuit Breakers with proper resets          |
//+------------------------------------------------------------------+
void UpdateCircuitBreakers()
{
    if(!InpUseCircuitBreaker && !InpUseMaxDD) return;

    MqlDateTime dt;
    TimeCurrent(dt);

    //--- Check for new day
    datetime today = StringToTime(IntegerToString(dt.year) + "." + IntegerToString(dt.mon) + "." + IntegerToString(dt.day));
    if(today != lastDayChecked)
    {
        dailyStartEquity = accInfo.Equity();
        lastDayChecked = today;
        dailyPaused = false; // Reset daily pause
        consecutiveLosses = 0; // FIX #2: Reset consecutive losses on new day
        Print("📅 New day - Daily equity reset to: ", DoubleToString(dailyStartEquity, 2));
        Print("   Consecutive losses reset to 0");
        SaveState();
    }

    //--- Check for new week (Monday)
    if(dt.day_of_week == 1)
    {
        datetime thisMonday = today;
        if(thisMonday != lastWeekChecked)
        {
            weeklyStartEquity = accInfo.Equity();
            lastWeekChecked = thisMonday;
            weeklyReduced = false; // Reset weekly reduction
            consecutiveLosses = 0; // FIX #2: Also reset on new week
            Print("📅 New week - Weekly equity reset to: ", DoubleToString(weeklyStartEquity, 2));
            Print("   Consecutive losses reset to 0");
            SaveState();
        }
    }

    //--- FIX #6: Max Drawdown Kill Switch (check FIRST before other breakers)
    if(InpUseMaxDD)
    {
        double equity = accInfo.Equity();
        if(equity > peakEquity)
        {
            peakEquity = equity;
            SaveState();
        }

        double dd = (peakEquity - equity) / peakEquity * 100;
        if(dd >= InpMaxDrawdownPct && !maxDDKilled)
        {
            maxDDKilled = true;
            dailyPaused = true;
            Print("💀💀💀 KILL SWITCH ACTIVATED 💀💀💀");
            Print("   Max Drawdown: ", DoubleToString(dd, 2), "% >= ", DoubleToString(InpMaxDrawdownPct, 1), "%");
            Print("   Peak Equity: ", DoubleToString(peakEquity, 2));
            Print("   Current Equity: ", DoubleToString(equity, 2));
            Print("   EA PERMANENTLY STOPPED - Manual intervention required");
            SaveState();
            return;
        }

        if(maxDDKilled)
        {
            dailyPaused = true;
            return;
        }
    }

    //--- Check daily loss limit
    double currentEquity = accInfo.Equity();
    double dailyPnL = ((currentEquity - dailyStartEquity) / dailyStartEquity) * 100;

    if(dailyPnL <= -InpDailyLossLimit)
    {
        if(!dailyPaused)
        {
            dailyPaused = true;
            Print("⛔ DAILY CIRCUIT BREAKER: Loss ", DoubleToString(dailyPnL, 2), "% >= ", InpDailyLossLimit, "%");
            Print("   Trading paused until tomorrow");
            SaveState();
        }
    }

    //--- Check weekly loss limit
    double weeklyPnL = ((currentEquity - weeklyStartEquity) / weeklyStartEquity) * 100;

    if(weeklyPnL <= -InpWeeklyLossLimit)
    {
        if(!weeklyReduced)
        {
            weeklyReduced = true;
            Print("⚠️ WEEKLY CIRCUIT BREAKER: Loss ", DoubleToString(weeklyPnL, 2), "% >= ", InpWeeklyLossLimit, "%");
            Print("   Position size reduced by 50% until next week");
            SaveState();
        }
    }

    //--- Check consecutive losses
    if(consecutiveLosses >= InpConsecLossLimit)
    {
        if(!dailyPaused)
        {
            dailyPaused = true;
            Print("⛔ CONSECUTIVE LOSSES CIRCUIT BREAKER: ", consecutiveLosses, " losses in a row");
            Print("   Trading paused for 24 hours");
            SaveState();
        }
    }
}

//+------------------------------------------------------------------+
//| Update Chart Comment                                              |
//+------------------------------------------------------------------+
void UpdateComment(double atr, double adx, int htfTrend)
{
    int positions = CountPositions();
    double equity = accInfo.Equity();
    double dailyPnL = dailyStartEquity > 0 ? ((equity - dailyStartEquity) / dailyStartEquity) * 100 : 0;
    double weeklyPnL = weeklyStartEquity > 0 ? ((equity - weeklyStartEquity) / weeklyStartEquity) * 100 : 0;
    double currentDD = peakEquity > 0 ? (peakEquity - equity) / peakEquity * 100 : 0;

    string htfStr = htfTrend > 0 ? "UP ↑" : (htfTrend < 0 ? "DOWN ↓" : "NEUTRAL ―");
    string skipStr = skipNextS1Signal ? "YES (waiting for S2 or loss)" : "NO";

    double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double spreadPct = atr > 0 ? (spread / atr) * 100 : 0;

    string comment = "═══════════════════════════════════\n";
    comment += "     🐢 TURTLE BOT V3 🐢\n";
    comment += "═══════════════════════════════════\n";
    comment += "\n📊 MARKET STATE\n";
    comment += "   ATR: " + DoubleToString(atr, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + "\n";
    comment += "   Spread: " + DoubleToString(spread, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) +
                 " (" + DoubleToString(spreadPct, 1) + "% ATR)" +
                 (spreadPct > InpMaxSpreadATR * 100 ? " ⚠️" : " ✅") + "\n";
    comment += "   ADX: " + DoubleToString(adx, 1) + (adx >= InpADXThreshold ? " ✅" : " ⏸️") + "\n";
    comment += "   HTF Trend: " + htfStr + "\n";
    comment += "   Session: " + (IsWithinSession() ? "ACTIVE ✅" : "CLOSED ⏸️") + "\n";
    comment += "\n📈 DONCHIAN CHANNELS\n";
    comment += "   S1 Entry (20): " + DoubleToString(highestHigh20, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + " / " + DoubleToString(lowestLow20, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + "\n";
    comment += "   S1 Exit (10): " + DoubleToString(highestHigh10, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + " / " + DoubleToString(lowestLow10, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + "\n";
    if(InpUseFailsafe)
        comment += "   S2 Failsafe (55): " + DoubleToString(highestHigh55, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + " / " + DoubleToString(lowestLow55, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + "\n";
    comment += "\n⚙️ RULES\n";
    comment += "   Skip Next S1: " + skipStr + "\n";
    comment += "   Consecutive Losses: " + IntegerToString(consecutiveLosses) + "/" + IntegerToString(InpConsecLossLimit) + "\n";
    comment += "\n💰 ACCOUNT\n";
    comment += "   Equity: $" + DoubleToString(equity, 2) + "\n";
    comment += "   Peak Equity: $" + DoubleToString(peakEquity, 2) + "\n";
    comment += "   Current DD: " + DoubleToString(currentDD, 2) + "% / " + DoubleToString(InpMaxDrawdownPct, 1) + "%\n";
    comment += "   Positions: " + IntegerToString(positions) + "/" + IntegerToString(InpMaxUnits) + "\n";
    comment += "   Daily P/L: " + DoubleToString(dailyPnL, 2) + "%\n";
    comment += "   Weekly P/L: " + DoubleToString(weeklyPnL, 2) + "%\n";
    comment += "\n🛡️ CIRCUIT BREAKERS\n";
    comment += "   Max DD Kill: " + (maxDDKilled ? "💀 KILLED" : "✅ OK") + "\n";
    comment += "   Daily Pause: " + (dailyPaused && !maxDDKilled ? "⛔ ACTIVE" : "✅ OK") + "\n";
    comment += "   Weekly Reduced: " + (weeklyReduced ? "⚠️ 50%" : "✅ 100%") + "\n";
    comment += "═══════════════════════════════════";

    Comment(comment);
}

//+------------------------------------------------------------------+
//| FIX #7: OnTrade - Process ALL Deals (not just last one)          |
//+------------------------------------------------------------------+
void OnTrade()
{
    static int lastDealsCount = 0;

    HistorySelect(TimeCurrent() - 86400, TimeCurrent());
    int dealsCount = HistoryDealsTotal();

    if(dealsCount <= lastDealsCount)
    {
        lastDealsCount = dealsCount;
        return;
    }

    double batchProfit = 0;
    bool hasClosedDeal = false;

    // Process ALL new deals since last check
    for(int i = lastDealsCount; i < dealsCount; i++)
    {
        ulong ticket = HistoryDealGetTicket(i);
        if(ticket <= 0) continue;
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != InpMagicNumber) continue;

        ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY);
        if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY)
        {
            // Accumulate total profit including commission and swap
            batchProfit += HistoryDealGetDouble(ticket, DEAL_PROFIT)
                         + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                         + HistoryDealGetDouble(ticket, DEAL_SWAP);
            hasClosedDeal = true;
        }
    }

    if(hasClosedDeal && InpUseSkipRule)
    {
        if(batchProfit > 0)
        {
            skipNextS1Signal = true;
            consecutiveLosses = 0;
            Print("📊 OnTrade: Batch closed with profit ", DoubleToString(batchProfit, 2), " → Skip next S1");
        }
        else
        {
            skipNextS1Signal = false;
            consecutiveLosses++;
            Print("📊 OnTrade: Batch closed with loss ", DoubleToString(batchProfit, 2), " → Take next S1 (Losses: ", consecutiveLosses, ")");
        }

        // Reset pyramid counter if all positions closed
        if(CountPositions() == 0)
        {
            currentUnits = 0;
            lastEntryPrice = 0;
            Print("   All positions closed, pyramid reset");
        }

        SaveState();
    }

    lastDealsCount = dealsCount;
}
//+------------------------------------------------------------------+
