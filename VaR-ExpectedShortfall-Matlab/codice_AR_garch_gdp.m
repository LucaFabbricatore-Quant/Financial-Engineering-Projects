% Lettura dei dati dal file CSV
andamento_prezzi_UNC = 'UniCredit Stock Price History.csv';
data = readtable(andamento_prezzi_UNC);

% Visualizzazione dei dati
disp(data);

% Calcolo dei daily log return
prezzi = data.Price;
logReturn = diff(log(prezzi));



% Grafico
figure;
plot(logReturn, '-o');
title('Log Return dei Prezzi di UniCredit');
xlabel('Giorni');
ylabel('Log Return');
grid on;

% Specifica il modello AR(1)-GARCH(1,1)
Mdl = arima('Constant', 0, 'ARLags', 1, 'Variance', garch(1,1));

% Stima dei parametri del modello AR(1)-GARCH(1,1)
EstMdl = estimate(Mdl, logReturn);

% Visualizzazione dei risultati della stima
disp(EstMdl);

% Verifica dell'adattamento del modello AR-GARCH ai dati iniziali
% Calcolo dei residui standardizzati
[innov, variance] = infer(EstMdl, logReturn);

% adattamento della lunghezza di variance
if length(innov) ~= length(variance)
    variance = variance(1:length(innov));
end

residui_standardizzati = innov ./ sqrt(variance);

%Test di Ljung-Box per verificare la presenza di autocorrelazione nei residui di un modello
[h_ar_garch, p_ar_garch] = lbqtest(residui_standardizzati);

disp('Test di Adattamento del Modello AR-GARCH ai Dati Iniziali:');
disp(['h = ', num2str(h_ar_garch), ', p = ', num2str(p_ar_garch)]);


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
title('Residui Standardizzati del Modello AR-GARCH');
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

% Previsioni a un passo avanti (t+1) per la media e la varianza condizionate
[Y, YMSE, V] = forecast(EstMdl, 1, 'Y0', logReturn);

% Visualizzazione delle previsioni della media e della varianza condizionate un passo avanti
disp('Previsione della Media Condizionata (t+1):');
disp(Y);

disp('Previsione della Varianza Condizionata (t+1):');
disp(V);

% Calcolo dei residui fino a t+1
residui = [innov; Y - mean(logReturn)];
variance = [variance; V];



% Calcolo della soglia per i residui rari 
soglia = quantile(residui_standardizzati, 0.95);

% Selezione dei residui che superano la soglia
residui_rari = residui_standardizzati(residui_standardizzati > soglia);

% Adattare una distribuzione di Pareto generalizzata (GPD) ai residui rari
parmHatGPD = mle(residui_rari, 'distribution', 'gp');

% Visualizzazione dei parametri della distribuzione di Pareto generalizzata
disp('Parametri della Distribuzione di Pareto Generalizzata:');
disp(parmHatGPD);

% Creazione della funzione CDF della GPD stimata
cdf_gpd = @(x) cdf('GeneralizedPareto', x, parmHatGPD(1), parmHatGPD(2), min(residui_rari));

% Verifica dell'adattamento della distribuzione di Pareto generalizzata ai residui rari
[h, p] = chi2gof(residui_rari, 'CDF', cdf_gpd);

disp('Test di Adattamento della Distribuzione di Pareto Generalizzata ai Residui Rari:');
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

% Grafico dei residui rari con sovrapposizione della distribuzione di Pareto generalizzata
figure;
histogram(residui_rari, 'Normalization', 'pdf');
hold on;
x = linspace(min(residui_rari), max(residui_rari), 100);
gpd_pdf = pdf(makedist('GeneralizedPareto', 'k', parmHatGPD(1), 'sigma', parmHatGPD(2), 'theta', min(residui_rari)), x);
plot(x, gpd_pdf, 'r', 'LineWidth', 2);
title('Istogramma dei Residui Rari e Distribuzione di Pareto Generalizzata');
xlabel('Residui Rari Standardizzati');
ylabel('Densità di Probabilità');
grid on;
legend('Residui Rari', 'Distribuzione di Pareto Generalizzata');

% impostare il livello di confidenza
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
