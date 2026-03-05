% Lettura dei dati dal file CSV
andamento_prezzi_UNC = 'UniCredit Stock Price History_per calcolo var.csv';
data = readtable(andamento_prezzi_UNC);

% Visualizzazione dei dati
disp(data);

% Calcolo dei daily log return
prezzi = data.Price;
logReturn = diff(log(prezzi));

% Visualizzare l'ultimo log-return disponibile
ultimo_logReturn = logReturn(end);
disp(['Ultimo log-return disponibile: ', num2str(ultimo_logReturn)]);

% Grafico
figure;
plot(logReturn, '-o');
title('Log Return dei Prezzi di UniCredit');
xlabel('Giorni');
ylabel('Log Return');
grid on;

% Specifica il modello ARMA(1,1)-EGARCH(1,1)
Mdl = arima('Constant', 0, 'ARLags', 1, 'MALags', 1, 'Variance', egarch(1,1));

% Stima dei parametri del modello ARMA(1,1)-EGARCH(1,1)
EstMdl = estimate(Mdl, logReturn);

% Visualizzazione dei risultati della stima
disp(EstMdl);

% Verifica dell'adattamento del modello ARMA-EGARCH ai dati iniziali
% Calcolo dei residui standardizzati
[innov, variance] = infer(EstMdl, logReturn);

% Adattamento della lunghezza di variance
if length(innov) ~= length(variance)
    variance = variance(1:length(innov));
end

residui_standardizzati = innov ./ sqrt(variance);

% Visualizzare l'ultima varianza disponibile
ultima_varianza = variance(end);
disp(['Ultima varianza disponibile: ', num2str(ultima_varianza)]);

% Test di Ljung-Box per verificare la presenza di autocorrelazione nei residui di un modello
[h_arma_egarch, p_arma_egarch] = lbqtest(residui_standardizzati);

disp('Test di Adattamento del Modello ARMA-EGARCH ai Dati Iniziali:');
disp(['h = ', num2str(h_arma_egarch), ', p = ', num2str(p_arma_egarch)]);

% Grafico dei daily log return con le varianze condizionali stimate
figure;
subplot(2,1,1);
plot(logReturn);
title('Daily Log Return');
xlabel('Giorni');
ylabel('Log Return');
grid on;

subplot(2,1,2);
plot(variance);
title('Varianze Condizionali Stimate');
xlabel('Giorni');
ylabel('Varianza Condizionale');
grid on;

% Grafico dei residui standardizzati
figure;
subplot(2,1,1);
plot(residui_standardizzati);
title('Residui Standardizzati del Modello ARMA-EGARCH');
xlabel('Giorni');
ylabel('Residui Standardizzati');
grid on;

% Istogramma dei residui standardizzati con sovrapposizione della distribuzione gaussiana
subplot(2,1,2);
histogram(residui_standardizzati, 'Normalization', 'pdf');
hold on;
x = linspace(min(residui_standardizzati), max(residui_standardizzati), 100);
y = normpdf(x, 0, 1);
plot(x, y, 'r', 'LineWidth', 2);
title('Istogramma dei Residui Standardizzati e Distribuzione Gaussiana');
xlabel('Residui Standardizzati');
ylabel('Densità di Probabilità');
grid on;
legend('Residui Standardizzati', 'Distribuzione Gaussiana');

% Calcolo dei residui fino a t
residui = innov;
variance = variance;

% Calcolo della soglia per i residui rari
soglia = quantile(residui_standardizzati, 0.95);

% Selezione dei residui che superano la soglia
residui_rari = residui_standardizzati(residui_standardizzati > soglia);

% Adattare una distribuzione t di Student ai residui rari
% La distribuzione t di Student con 1 grado di libertà corrisponde alla distribuzione Cauchy
parmHatT = mle(residui_rari, 'distribution', 'tlocationscale');

% Visualizzazione dei parametri della distribuzione t di Student
disp('Parametri della Distribuzione t di Student:');
disp(parmHatT);

% Creazione della funzione CDF della distribuzione t di Student stimata
cdf_t = @(x) tcdf((x - parmHatT(1)) / parmHatT(2), 1);

% Verifica dell'adattamento della distribuzione t di Student ai residui rari
[h, p] = chi2gof(residui_rari, 'CDF', cdf_t);

disp('Test di Adattamento della Distribuzione t di Student ai Residui Rari:');
disp(['h = ', num2str(h), ', p = ', num2str(p)]);

% Grafico dei residui e della soglia
figure;
histogram(residui_standardizzati, 30, 'Normalization', 'pdf');
hold on;
xline(soglia, 'r', 'LineWidth', 2);
title('Distribuzione dei Residui e Soglia per i Residui Rari');
xlabel('Residui Standardizzati');
ylabel('Densità di Probabilità');
grid on;
legend('Residui', 'Soglia 95%');

% Grafico dei residui rari con sovrapposizione della distribuzione t di Student
figure;
histogram(residui_rari, 'Normalization', 'pdf');
hold on;
x = linspace(min(residui_rari), max(residui_rari), 100);
t_pdf = tpdf((x - parmHatT(1)) / parmHatT(2), 1) / parmHatT(2);
plot(x, t_pdf, 'r', 'LineWidth', 2);
title('Istogramma dei Residui Rari e Distribuzione t di Student');
xlabel('Residui Rari Standardizzati');
ylabel('Densità di Probabilità');
grid on;
legend('Residui Rari', 'Distribuzione t di Student');

% Impostare il livello di confidenza
confidenza = 0.95;

% Ordinare i residui standardizzati in ordine crescente
residui_ordinati = sort(residui_standardizzati);

% Calcolo del Value at Risk (VaR)
indice_var = round((1 - confidenza) * length(residui_ordinati));
VaR = residui_ordinati(indice_var);

% Calcolo dell'Expected Shortfall (ES)
ES = mean(residui_ordinati(1:indice_var));

% Visualizzazione dei risultati
disp(['Value at Risk (VaR) a ', num2str(confidenza * 100), '% livello di confidenza: ', num2str(VaR)]);
disp(['Expected Shortfall (ES) a ', num2str(confidenza * 100), '% livello di confidenza: ', num2str(ES)]);
