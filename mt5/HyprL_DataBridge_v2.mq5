//+------------------------------------------------------------------+
//|                                        HyprL_DataBridge_v2.mq5   |
//|                              File-based bridge for Python        |
//+------------------------------------------------------------------+
#property copyright "HyprL"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   DataFolder = "HyprL";           // Folder for data exchange
input int      MaxBars = 50000;                // Max bars to export
input int      CheckIntervalMs = 500;          // Check interval (ms)

//--- Global
CTrade trade;
string dataPath;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
    // Create data folder
    dataPath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + DataFolder;

    // Use timer for checking commands
    EventSetMillisecondTimer(CheckIntervalMs);

    Print("HyprL DataBridge v2 started");
    Print("Data folder: ", dataPath);
    Print("Drop command files in: MQL5\\Files\\", DataFolder, "\\");

    // Write initial status
    WriteStatus("ready");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    WriteStatus("stopped");
    Print("HyprL DataBridge stopped");
}

//+------------------------------------------------------------------+
//| Timer event - check for commands                                  |
//+------------------------------------------------------------------+
void OnTimer()
{
    CheckCommands();
}

//+------------------------------------------------------------------+
//| Check for command files                                           |
//+------------------------------------------------------------------+
void CheckCommands()
{
    // Check for export command: export_SYMBOL_TIMEFRAME.cmd
    string cmdFile = DataFolder + "\\command.txt";

    if(FileIsExist(cmdFile))
    {
        int handle = FileOpen(cmdFile, FILE_READ|FILE_TXT);
        if(handle != INVALID_HANDLE)
        {
            string command = FileReadString(handle);
            FileClose(handle);
            FileDelete(cmdFile);

            ProcessCommand(command);
        }
    }
}

//+------------------------------------------------------------------+
//| Process command                                                   |
//+------------------------------------------------------------------+
void ProcessCommand(string command)
{
    Print("Processing command: ", command);

    string parts[];
    int count = StringSplit(command, '|', parts);

    if(count < 1) return;

    string action = parts[0];

    if(action == "EXPORT" && count >= 4)
    {
        // EXPORT|SYMBOL|TIMEFRAME|BARS
        string symbol = parts[1];
        string tf = parts[2];
        int bars = (int)StringToInteger(parts[3]);
        ExportBars(symbol, tf, bars);
    }
    else if(action == "ACCOUNT")
    {
        ExportAccount();
    }
    else if(action == "POSITIONS")
    {
        ExportPositions();
    }
    else if(action == "SYMBOLS")
    {
        ExportSymbols();
    }
    else if(action == "ORDER" && count >= 5)
    {
        // ORDER|SYMBOL|ACTION|VOLUME|SL|TP
        string symbol = parts[1];
        string act = parts[2];
        double volume = StringToDouble(parts[3]);
        double sl = count > 4 ? StringToDouble(parts[4]) : 0;
        double tp = count > 5 ? StringToDouble(parts[5]) : 0;
        ExecuteOrder(symbol, act, volume, sl, tp);
    }
    else
    {
        WriteResult("error", "Unknown command: " + command);
    }
}

//+------------------------------------------------------------------+
//| Export bars to CSV                                                |
//+------------------------------------------------------------------+
void ExportBars(string symbol, string tfStr, int count)
{
    ENUM_TIMEFRAMES tf = StringToTimeframe(tfStr);

    if(!SymbolSelect(symbol, true))
    {
        WriteResult("error", "Symbol not found: " + symbol);
        return;
    }

    if(count > MaxBars) count = MaxBars;

    MqlRates rates[];
    int copied = CopyRates(symbol, tf, 0, count, rates);

    if(copied <= 0)
    {
        WriteResult("error", "Failed to copy rates for " + symbol);
        return;
    }

    // Write to CSV
    string filename = DataFolder + "\\" + symbol + "_" + tfStr + ".csv";
    int handle = FileOpen(filename, FILE_WRITE|FILE_CSV);

    if(handle == INVALID_HANDLE)
    {
        WriteResult("error", "Cannot create file: " + filename);
        return;
    }

    // Header
    FileWrite(handle, "time", "open", "high", "low", "close", "volume");

    // Data
    for(int i = 0; i < copied; i++)
    {
        FileWrite(handle,
            (long)rates[i].time,
            rates[i].open,
            rates[i].high,
            rates[i].low,
            rates[i].close,
            rates[i].tick_volume
        );
    }

    FileClose(handle);

    WriteResult("ok", "Exported " + IntegerToString(copied) + " bars to " + filename);
    Print("Exported ", copied, " bars for ", symbol, " ", tfStr);
}

//+------------------------------------------------------------------+
//| Export account info                                               |
//+------------------------------------------------------------------+
void ExportAccount()
{
    string filename = DataFolder + "\\account.csv";
    int handle = FileOpen(filename, FILE_WRITE|FILE_CSV);

    if(handle == INVALID_HANDLE)
    {
        WriteResult("error", "Cannot create account file");
        return;
    }

    FileWrite(handle, "field", "value");
    FileWrite(handle, "balance", AccountInfoDouble(ACCOUNT_BALANCE));
    FileWrite(handle, "equity", AccountInfoDouble(ACCOUNT_EQUITY));
    FileWrite(handle, "margin", AccountInfoDouble(ACCOUNT_MARGIN));
    FileWrite(handle, "free_margin", AccountInfoDouble(ACCOUNT_MARGIN_FREE));
    FileWrite(handle, "leverage", AccountInfoInteger(ACCOUNT_LEVERAGE));
    FileWrite(handle, "currency", AccountInfoString(ACCOUNT_CURRENCY));
    FileWrite(handle, "name", AccountInfoString(ACCOUNT_NAME));
    FileWrite(handle, "server", AccountInfoString(ACCOUNT_SERVER));

    FileClose(handle);
    WriteResult("ok", "Account exported");
}

//+------------------------------------------------------------------+
//| Export positions                                                  |
//+------------------------------------------------------------------+
void ExportPositions()
{
    string filename = DataFolder + "\\positions.csv";
    int handle = FileOpen(filename, FILE_WRITE|FILE_CSV);

    if(handle == INVALID_HANDLE)
    {
        WriteResult("error", "Cannot create positions file");
        return;
    }

    FileWrite(handle, "ticket", "symbol", "type", "volume", "open_price", "current_price", "profit", "sl", "tp");

    int total = PositionsTotal();
    for(int i = 0; i < total; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0)
        {
            FileWrite(handle,
                ticket,
                PositionGetString(POSITION_SYMBOL),
                PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "buy" : "sell",
                PositionGetDouble(POSITION_VOLUME),
                PositionGetDouble(POSITION_PRICE_OPEN),
                PositionGetDouble(POSITION_PRICE_CURRENT),
                PositionGetDouble(POSITION_PROFIT),
                PositionGetDouble(POSITION_SL),
                PositionGetDouble(POSITION_TP)
            );
        }
    }

    FileClose(handle);
    WriteResult("ok", "Positions exported: " + IntegerToString(total));
}

//+------------------------------------------------------------------+
//| Export available symbols                                          |
//+------------------------------------------------------------------+
void ExportSymbols()
{
    string filename = DataFolder + "\\symbols.csv";
    int handle = FileOpen(filename, FILE_WRITE|FILE_CSV);

    if(handle == INVALID_HANDLE)
    {
        WriteResult("error", "Cannot create symbols file");
        return;
    }

    FileWrite(handle, "symbol", "digits", "trade_mode");

    int total = SymbolsTotal(false);
    int exported = 0;

    for(int i = 0; i < total; i++)
    {
        string name = SymbolName(i, false);
        long mode = SymbolInfoInteger(name, SYMBOL_TRADE_MODE);

        if(mode != SYMBOL_TRADE_MODE_DISABLED)
        {
            FileWrite(handle, name, SymbolInfoInteger(name, SYMBOL_DIGITS), mode);
            exported++;
        }
    }

    FileClose(handle);
    WriteResult("ok", "Symbols exported: " + IntegerToString(exported));
}

//+------------------------------------------------------------------+
//| Execute order                                                     |
//+------------------------------------------------------------------+
void ExecuteOrder(string symbol, string action, double volume, double sl, double tp)
{
    trade.SetExpertMagicNumber(123456);
    trade.SetDeviationInPoints(10);

    bool result = false;

    if(action == "BUY")
    {
        double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
        result = trade.Buy(volume, symbol, ask, sl, tp, "HyprL");
    }
    else if(action == "SELL")
    {
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        result = trade.Sell(volume, symbol, bid, sl, tp, "HyprL");
    }
    else if(action == "CLOSE")
    {
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == symbol)
            {
                trade.PositionClose(ticket);
            }
        }
        result = true;
    }

    if(result)
    {
        WriteResult("ok", "Order executed: " + action + " " + symbol);
    }
    else
    {
        WriteResult("error", "Order failed: " + trade.ResultRetcodeDescription());
    }
}

//+------------------------------------------------------------------+
//| String to timeframe                                               |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tf)
{
    if(tf == "M1" || tf == "1m") return PERIOD_M1;
    if(tf == "M5" || tf == "5m") return PERIOD_M5;
    if(tf == "M15" || tf == "15m") return PERIOD_M15;
    if(tf == "M30" || tf == "30m") return PERIOD_M30;
    if(tf == "H1" || tf == "1h") return PERIOD_H1;
    if(tf == "H4" || tf == "4h") return PERIOD_H4;
    if(tf == "D1" || tf == "1d") return PERIOD_D1;
    if(tf == "W1" || tf == "1w") return PERIOD_W1;
    return PERIOD_H1;
}

//+------------------------------------------------------------------+
//| Write status file                                                 |
//+------------------------------------------------------------------+
void WriteStatus(string status)
{
    string filename = DataFolder + "\\status.txt";
    int handle = FileOpen(filename, FILE_WRITE|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, status);
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Write result file                                                 |
//+------------------------------------------------------------------+
void WriteResult(string status, string message)
{
    string filename = DataFolder + "\\result.txt";
    int handle = FileOpen(filename, FILE_WRITE|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, status + "|" + message);
        FileClose(handle);
    }
}
//+------------------------------------------------------------------+
