//+------------------------------------------------------------------+
//|                                               HyprL_Bridge.mq5   |
//|                                    HyprL Algo Trading Bridge     |
//|                                         https://hyprlcore.com    |
//+------------------------------------------------------------------+
#property copyright "HyprL"
#property link      "https://hyprlcore.com"
#property version   "1.00"
#property strict

// Input parameters
input string   API_URL = "https://hyprlcore.com/mt5-api";  // HyprL API URL
input string   API_KEY = "hyprl_mt5_ftmo_2026";  // API Key
input int      POLL_SECONDS = 60;          // Poll interval (seconds)
input double   RISK_PERCENT = 1.0;         // Risk per trade (%)
input double   LOT_SIZE = 0.1;             // Fixed lot size (if risk calc fails)
input int      SLIPPAGE = 30;              // Max slippage (points)
input int      MAGIC_NUMBER = 123456;      // EA Magic Number
input bool     ENABLE_TRADING = true;      // Enable live trading
input bool     DEBUG_MODE = true;          // Debug mode

// Global variables
datetime lastPollTime = 0;
string lastSignals = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("HyprL Bridge EA initialized");
   Print("API URL: ", API_URL);
   Print("Poll interval: ", POLL_SECONDS, " seconds");
   Print("Risk: ", RISK_PERCENT, "%");
   Print("Trading enabled: ", ENABLE_TRADING);

   // Initial poll
   PollSignals();

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("HyprL Bridge EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if it's time to poll
   if(TimeCurrent() - lastPollTime >= POLL_SECONDS)
   {
      PollSignals();
      lastPollTime = TimeCurrent();
   }
}

//+------------------------------------------------------------------+
//| Poll API for signals                                               |
//+------------------------------------------------------------------+
void PollSignals()
{
   string url = API_URL + "/signals?key=" + API_KEY;
   string headers = "Content-Type: application/json\r\n";
   char post[];
   char result[];
   string resultHeaders;

   if(DEBUG_MODE) Print("Polling: ", url);

   int timeout = 5000; // 5 seconds
   int res = WebRequest("GET", url, headers, timeout, post, result, resultHeaders);

   if(res == -1)
   {
      int error = GetLastError();
      Print("WebRequest error: ", error);
      if(error == 4014)
      {
         Print("Add URL to allowed list: Tools -> Options -> Expert Advisors -> Allow WebRequest for listed URL");
         Print("Add: ", API_URL);
      }
      return;
   }

   string response = CharArrayToString(result);

   if(DEBUG_MODE) Print("Response: ", StringSubstr(response, 0, 200));

   // Parse JSON response
   ProcessSignals(response);
}

//+------------------------------------------------------------------+
//| Process signals from API response                                  |
//+------------------------------------------------------------------+
void ProcessSignals(string json)
{
   // Simple JSON parsing (MT5 doesn't have native JSON support)
   // Looking for pattern: "signals":[{...},{...}]

   int signalsStart = StringFind(json, "\"signals\":[");
   if(signalsStart == -1)
   {
      if(DEBUG_MODE) Print("No signals array found");
      return;
   }

   // Extract signals array content
   int arrayStart = StringFind(json, "[", signalsStart);
   int arrayEnd = StringFind(json, "]", arrayStart);

   if(arrayStart == -1 || arrayEnd == -1)
   {
      Print("Invalid signals format");
      return;
   }

   string signalsContent = StringSubstr(json, arrayStart + 1, arrayEnd - arrayStart - 1);

   // Skip if same as last poll (no changes)
   if(signalsContent == lastSignals)
   {
      if(DEBUG_MODE) Print("No signal changes");
      return;
   }
   lastSignals = signalsContent;

   // Parse individual signals
   // Format: {"symbol":"NVDA","mt5_symbol":"NVDA.US","direction":"long",...}

   int pos = 0;
   while(pos < StringLen(signalsContent))
   {
      int objStart = StringFind(signalsContent, "{", pos);
      if(objStart == -1) break;

      int objEnd = StringFind(signalsContent, "}", objStart);
      if(objEnd == -1) break;

      string signalObj = StringSubstr(signalsContent, objStart, objEnd - objStart + 1);
      ProcessSingleSignal(signalObj);

      pos = objEnd + 1;
   }
}

//+------------------------------------------------------------------+
//| Process a single signal                                            |
//+------------------------------------------------------------------+
void ProcessSingleSignal(string signalJson)
{
   // Extract fields from JSON object
   string mt5Symbol = ExtractJsonString(signalJson, "mt5_symbol");
   string direction = ExtractJsonString(signalJson, "direction");
   double stopLoss = ExtractJsonDouble(signalJson, "stop_loss");
   double takeProfit = ExtractJsonDouble(signalJson, "take_profit");
   double probability = ExtractJsonDouble(signalJson, "probability");

   if(mt5Symbol == "" || direction == "")
   {
      Print("Invalid signal: missing symbol or direction");
      return;
   }

   Print("Signal: ", mt5Symbol, " ", direction, " SL=", stopLoss, " TP=", takeProfit, " Prob=", probability);

   // Check if symbol exists
   if(!SymbolSelect(mt5Symbol, true))
   {
      Print("Symbol not found: ", mt5Symbol);
      return;
   }

   // Execute signal
   if(direction == "long")
   {
      ExecuteLong(mt5Symbol, stopLoss, takeProfit);
   }
   else if(direction == "short")
   {
      ExecuteShort(mt5Symbol, stopLoss, takeProfit);
   }
   else if(direction == "flat")
   {
      ClosePosition(mt5Symbol);
   }
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                     |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
   string searchKey = "\"" + key + "\":\"";
   int start = StringFind(json, searchKey);
   if(start == -1) return "";

   start += StringLen(searchKey);
   int end = StringFind(json, "\"", start);
   if(end == -1) return "";

   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extract double value from JSON                                     |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int start = StringFind(json, searchKey);
   if(start == -1) return 0;

   start += StringLen(searchKey);

   // Find end of number (comma, }, or end of string)
   int end = start;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ' ' || ch == '\n') break;
      end++;
   }

   string valueStr = StringSubstr(json, start, end - start);
   return StringToDouble(valueStr);
}

//+------------------------------------------------------------------+
//| Check if position exists for symbol                                |
//+------------------------------------------------------------------+
bool HasPosition(string symbol)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Get position type for symbol                                       |
//+------------------------------------------------------------------+
ENUM_POSITION_TYPE GetPositionType(string symbol)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER)
            return (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      }
   }
   return (ENUM_POSITION_TYPE)-1;
}

//+------------------------------------------------------------------+
//| Execute Long Trade                                                 |
//+------------------------------------------------------------------+
void ExecuteLong(string symbol, double sl, double tp)
{
   if(!ENABLE_TRADING)
   {
      Print("[DRY RUN] Would BUY ", symbol, " SL=", sl, " TP=", tp);
      return;
   }

   // Check existing position
   if(HasPosition(symbol))
   {
      ENUM_POSITION_TYPE posType = GetPositionType(symbol);
      if(posType == POSITION_TYPE_BUY)
      {
         Print("Already LONG ", symbol);
         return;
      }
      else if(posType == POSITION_TYPE_SELL)
      {
         Print("Closing SHORT before opening LONG ", symbol);
         ClosePosition(symbol);
      }
   }

   // Calculate lot size
   double lots = CalculateLotSize(symbol, sl);
   if(lots <= 0) lots = LOT_SIZE;

   // Get current price
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);

   // Adjust SL/TP to price levels
   if(sl > 0 && sl > ask) sl = ask - (ask * 0.03); // Default 3% SL
   if(tp > 0 && tp < ask) tp = ask + (ask * 0.06); // Default 6% TP

   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = lots;
   request.type = ORDER_TYPE_BUY;
   request.price = ask;
   request.sl = sl;
   request.tp = tp;
   request.deviation = SLIPPAGE;
   request.magic = MAGIC_NUMBER;
   request.comment = "HyprL Long";
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result))
   {
      Print("OrderSend error: ", GetLastError());
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE)
   {
      Print("BUY executed: ", symbol, " @ ", result.price, " Lots: ", lots);
   }
   else
   {
      Print("BUY failed: ", result.retcode, " ", result.comment);
   }
}

//+------------------------------------------------------------------+
//| Execute Short Trade                                                |
//+------------------------------------------------------------------+
void ExecuteShort(string symbol, double sl, double tp)
{
   if(!ENABLE_TRADING)
   {
      Print("[DRY RUN] Would SELL ", symbol, " SL=", sl, " TP=", tp);
      return;
   }

   // Check existing position
   if(HasPosition(symbol))
   {
      ENUM_POSITION_TYPE posType = GetPositionType(symbol);
      if(posType == POSITION_TYPE_SELL)
      {
         Print("Already SHORT ", symbol);
         return;
      }
      else if(posType == POSITION_TYPE_BUY)
      {
         Print("Closing LONG before opening SHORT ", symbol);
         ClosePosition(symbol);
      }
   }

   // Calculate lot size
   double lots = CalculateLotSize(symbol, sl);
   if(lots <= 0) lots = LOT_SIZE;

   // Get current price
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);

   // Adjust SL/TP to price levels
   if(sl > 0 && sl < bid) sl = bid + (bid * 0.03); // Default 3% SL
   if(tp > 0 && tp > bid) tp = bid - (bid * 0.06); // Default 6% TP

   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = lots;
   request.type = ORDER_TYPE_SELL;
   request.price = bid;
   request.sl = sl;
   request.tp = tp;
   request.deviation = SLIPPAGE;
   request.magic = MAGIC_NUMBER;
   request.comment = "HyprL Short";
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result))
   {
      Print("OrderSend error: ", GetLastError());
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE)
   {
      Print("SELL executed: ", symbol, " @ ", result.price, " Lots: ", lots);
   }
   else
   {
      Print("SELL failed: ", result.retcode, " ", result.comment);
   }
}

//+------------------------------------------------------------------+
//| Close position for symbol                                          |
//+------------------------------------------------------------------+
void ClosePosition(string symbol)
{
   if(!ENABLE_TRADING)
   {
      Print("[DRY RUN] Would close ", symbol);
      return;
   }

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER)
         {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            double volume = PositionGetDouble(POSITION_VOLUME);
            ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

            MqlTradeRequest request = {};
            MqlTradeResult result = {};

            request.action = TRADE_ACTION_DEAL;
            request.symbol = symbol;
            request.volume = volume;
            request.position = ticket;
            request.deviation = SLIPPAGE;
            request.magic = MAGIC_NUMBER;
            request.comment = "HyprL Close";
            request.type_filling = ORDER_FILLING_IOC;

            if(type == POSITION_TYPE_BUY)
            {
               request.type = ORDER_TYPE_SELL;
               request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
            }
            else
            {
               request.type = ORDER_TYPE_BUY;
               request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
            }

            if(!OrderSend(request, result))
            {
               Print("Close error: ", GetLastError());
            }
            else if(result.retcode == TRADE_RETCODE_DONE)
            {
               Print("Position closed: ", symbol);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                   |
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double stopLoss)
{
   if(stopLoss <= 0) return LOT_SIZE;

   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * (RISK_PERCENT / 100.0);

   double price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(tickValue <= 0 || tickSize <= 0) return LOT_SIZE;

   double slDistance = MathAbs(price - stopLoss);
   double slTicks = slDistance / tickSize;

   if(slTicks <= 0) return LOT_SIZE;

   double lots = riskAmount / (slTicks * tickValue);

   // Normalize to lot step
   lots = MathFloor(lots / lotStep) * lotStep;

   // Clamp to min/max
   if(lots < minLot) lots = minLot;
   if(lots > maxLot) lots = maxLot;

   return lots;
}

//+------------------------------------------------------------------+
//| Timer function (alternative to OnTick for testing)                 |
//+------------------------------------------------------------------+
void OnTimer()
{
   PollSignals();
}
//+------------------------------------------------------------------+
