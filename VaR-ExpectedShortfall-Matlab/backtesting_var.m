% Lettura dei dati dal file CSV
andamento_prezzi_UNC = 'UniCredit Stock Price History.csv';
data = readtable(andamento_prezzi_UNC);

% Dati storici dei log return
prezzi = data.Price;
logReturn = diff(log(prezzi));

% Numero di giorni per il backtesting
num_backtest = 1000;
violazioni = 0;

% Livello di confidenza per il VaR
livello_confidenza = 0.95;

for i = 1:num_backtest
    VaR_t = quantile(logReturn(i:end), 1 - livello_confidenza); % Calcola il VaR a 95% di confidenza
    if logReturn(i) < VaR_t
        violazioni = violazioni + 1;
    end
end

tasso_violazione = violazioni / num_backtest;
disp(['Tasso di violazione del VaR: ', num2str(tasso_violazione)]);
