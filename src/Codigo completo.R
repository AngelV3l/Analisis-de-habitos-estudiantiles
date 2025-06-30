#==================================================
# CARGA DE PAQUETES
#==================================================
library(psych)
library(psychTools)
library(ggfortify)
library(ggplot2)
library(car)
library(MASS) 
library(leaps)
library(factoextra)
library(cluster)
library(clusterSim)
library(dendextend)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(gbm)

#==================================================
# CARGA Y LIMPIEZA DE DATOS
#==================================================

# Cargar datos
datos <- read.table("student_habits_performance.csv", sep = ",", dec = ".", header = TRUE)

# Vista preliminar
head(datos)

# Comprobación de valores faltantes
sum(is.na(datos))  # No hay valores faltantes

# Eliminar columna de ID (irrelevante para el análisis)
datos <- datos[, -1]

# Renombrar columnas a español
colnames(datos) <- c("Edad", "Sexo", "Horas_estudio_diario", "Horas_redes_sociales", "Horas_netflix",
                     "Trabajo", "Asistencia", "Horas_sueño", "Dieta", "Dias_ejercicio_semanal",
                     "Educacion_padres", "Calidad_internet", "Salud_mental",
                     "Actividades_extracurriculares", "Calificacion")

# Conversión de variables character a factor
str(datos)
datos$Sexo <- as.factor(datos$Sexo)
datos$Trabajo <- as.factor(datos$Trabajo)
datos$Dieta <- as.factor(datos$Dieta)
datos$Educacion_padres <- as.factor(datos$Educacion_padres)
datos$Calidad_internet <- as.factor(datos$Calidad_internet)
datos$Actividades_extracurriculares <- as.factor(datos$Actividades_extracurriculares)

# Escalar variables a formatos adecuados
datos$Calificacion <- datos$Calificacion / 10  # De 0-100 a escala 0-10
datos$Asistencia   <- datos$Asistencia / 100   # De porcentaje a proporción (0-1)

#==================================================
# ANÁLISIS DESCRIPTIVO
#==================================================

# Estadísticos descriptivos básicos
describe(datos)

# Boxplots para detección visual de outliers
par(mfrow = c(3, 3))
boxplot(datos$Edad, main = "Edad")
boxplot(datos$Horas_estudio_diario, main = "Horas de Estudio Diario")
boxplot(datos$Horas_redes_sociales, main = "Horas en Redes Sociales")
boxplot(datos$Horas_netflix, main = "Horas en Netflix")
boxplot(datos$Asistencia, main = "Asistencia")
boxplot(datos$Horas_sueño, main = "Horas de Sueño")
boxplot(datos$Dias_ejercicio_semanal, main = "Días de Ejercicio Semanal")
boxplot(datos$Salud_mental, main = "Salud Mental")
boxplot(datos$Calificacion, main = "Calificación")

# Histogramas para distribución de variables numéricas
par(mfrow = c(3, 3))
hist(datos$Edad, main = "Histograma de Edad", xlab = "Edad", col = "lightblue", border = "black")
hist(datos$Horas_estudio_diario, main = "Histograma de Horas Estudio", xlab = "Horas", col = "lightblue", border = "black")
hist(datos$Horas_redes_sociales, main = "Histograma de Redes Sociales", xlab = "Horas", col = "lightblue", border = "black")
hist(datos$Horas_netflix, main = "Histograma de Netflix", xlab = "Horas", col = "lightblue", border = "black")
hist(datos$Asistencia, main = "Histograma de Asistencia", xlab = "Asistencia", col = "lightblue", border = "black")
hist(datos$Horas_sueño, main = "Histograma de Horas de Sueño", xlab = "Horas", col = "lightblue", border = "black")
hist(datos$Dias_ejercicio_semanal, main = "Histograma de Ejercicio", xlab = "Días", col = "lightblue", border = "black")
hist(datos$Salud_mental, main = "Histograma de Salud Mental", xlab = "Puntuación", col = "lightblue", border = "black")
hist(datos$Calificacion, main = "Histograma de Calificación", xlab = "Calificación", col = "lightblue", border = "black")

# Gráficos de barras para variables categóricas
par(mfrow = c(2, 3))
barplot(table(datos$Sexo), main = "Sexo", col = "lightgreen", border = "black")
barplot(table(datos$Trabajo), main = "Trabajo", col = "lightgreen", border = "black")
barplot(table(datos$Dieta), main = "Dieta", col = "lightgreen", border = "black")
barplot(table(datos$Educacion_padres), main = "Nivel de Educación de los Padres", 
        col = "lightgreen", border = "black", las = 2)
barplot(table(datos$Calidad_internet), main = "Calidad de Internet", 
        col = "lightgreen", border = "black", las = 2)
barplot(table(datos$Actividades_extracurriculares), main = "Participación en Act. Extracurriculares", 
        col = "lightgreen", border = "black")

#==================================================
# MATRIZ DE CORRELACIONES
#==================================================

# Matriz de correlaciones entre variables numéricas
cor_matrix <- cor(datos[, sapply(datos, is.numeric)])

# Visualización de correlaciones
corPlot(cor_matrix, numbers = TRUE, upper = FALSE, xlas = 2)



#==================================================
# REGRESIÓN LINEAL MÚLTIPLE
#==================================================

# División de los datos en entrenamiento y prueba
set.seed(1009)
train <- sample(nrow(datos), 0.7 * nrow(datos))
test  <- datos[-train, ]

# Modelo inicial con todas las variables
modelo1 <- lm(Calificacion ~ ., data = datos[train, ])
summary(modelo1)


### DIAGNÓSTICO DEL MODELO INICIAL

# Evaluación global del modelo (normalidad de los residuos y colinealidad)
check_model(modelo1, check = c("vif", "normality"))
vif(modelo1)

# Gráficos de diagnóstico
autoplot(modelo1, which = 1:4)


### EVALUACIÓN PREDICTIVA DEL MODELO INICIAL

# Predicciones sobre el conjunto de prueba
predicciones <- predict(modelo1, test)

# Gráfico de predicciones vs valores reales
par(mfrow=c(1,1))
plot(predicciones, test$Calificacion, pch = 20,
     xlab = "Predicciones", ylab = "Calificación real",
     main = "Predicciones vs Valores reales")

# Métricas de rendimiento
cor(predicciones, test$Calificacion)
model_performance(modelo1)


### MODELO REDUCIDO: SELECCIÓN AUTOMÁTICA

# Modelo reducido mediante stepwise (stepAIC)
modelo.red <- stepAIC(modelo1, direction = "both", trace = TRUE)
summary(modelo.red)

# Evaluación global del modelo (normalidad de los residuos y colinealidad)
check_model(modelo.red, check = c("vif", "normality"))
vif(modelo.red)

# Gráficos de diagnóstico
autoplot(modelo.red, which = 1:4)

# Evaluación del modelo reducido
predicciones.red <- predict(modelo.red, test)

# Gráfico de predicciones vs valores reales (modelo reducido)
plot(predicciones.red, test$Calificacion, pch = 20,
     xlab = "Predicciones", ylab = "Calificación real",
     main = "Predicciones vs Valores reales (modelo reducido)")

# Métricas de rendimiento
cor(predicciones.red, test$Calificacion)
model_performance(modelo.red)

#Comparación de los modelos
compare_performance(modelo1, modelo.red, verbose = FALSE)



### MODELO REDUCIDO (SELECCIÓN POR SUBCONJUNTOS)

# Aplicación del algoritmo
regfit <- regsubsets(Calificacion ~ ., data = datos[train, ])
res <- summary(regfit)
res

# Gráficos: evolución del R² ajustado y BIC
par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
plot(1:8, res$adjr2, type = "l", xlab = "Número de parámetros", ylab = "R² ajustado", main = "R² ajustado")
plot(1:8, res$bic, type = "l", xlab = "Número de parámetros", ylab = "BIC", main = "Bayesian Information Criterion")

# Variables seleccionadas en el mejor modelo (según R² ajustado)
mejor.r <- which.max(res$adjr2)
res$adjr2[mejor.r]
coef(regfit, mejor.r)

# Construcción del modelo con las variables seleccionadas
modelo.sub <- lm(Calificacion ~ Horas_estudio_diario +
                   Horas_redes_sociales +
                   Horas_netflix +
                   Asistencia +
                   Horas_sueño +
                   Dias_ejercicio_semanal +
                   Calidad_internet +
                   Salud_mental,
                 data = datos[train, ])
summary(modelo.sub)  

# Diagnóstico del modelo por subconjuntos
check_model(modelo.sub, check = c("vif", "normality"))
vif(modelo.sub)
autoplot(modelo.sub, which = 1:4)

# Evaluación predictiva
predicciones.sub <- predict(modelo.sub, test)

# Gráfico de dispersión: predicciones vs reales
par(mfrow=c(1,1))
plot(predicciones.sub, test$Calificacion, pch = 20,
     xlab = "Predicciones", ylab = "Calificación real",
     main = "Predicciones vs Valores reales")

# Métricas de rendimiento
cor(predicciones.sub, test$Calificacion)
model_performance(modelo.sub)


### COMPARACIÓN DE LOS 3 MODELOS

compare_performance(modelo1, modelo.red, modelo.sub, rank = TRUE)



#==================================================
# REDUCCIÓN DE LA DIMENSIONALIDAD
#==================================================

#==================================================
# ANÁLISIS DE COMPONENTES PRINCIPALES (ACP)
#==================================================

# Seleccionar solo las variables numéricas
vars_numericas <- datos[, c("Edad", "Horas_estudio_diario", "Horas_redes_sociales",
                            "Horas_netflix", "Asistencia", "Horas_sueño",
                            "Dias_ejercicio_semanal", "Salud_mental", "Calificacion")]

# Aplicar ACP usando la matriz de correlaciones (equivale a estandarizar)
CP <- princomp(vars_numericas, cor = TRUE)

# Resumen de componentes y varianza explicada
summary(CP)

# Scree plot: autovalores de los componentes
fviz_eig(CP, addlabels = TRUE)

#No es aplicable el ACP

#==================================================
# ANÁLISIS FACTORIAL
#==================================================

#Determinante de la matriz de correlación
det(cor_matrix)

#Medida de kaiser-M-O
print(KMO(cor_matrix),digits=4)

#No es aplicable el análisis factorial



#==================================================
# CLASIFICACIÓN
#==================================================

#==================================================
# CLASIFICACIÓN NO SUPERVISADA
#==================================================

### PREPARACIÓN DE LOS DATOS

# Selección de variables numéricas relevantes (comentar una de las dos)
#  datos_clasif <- datos[, c("Horas_estudio_diario",
                           # "Horas_redes_sociales",
                           # "Horas_netflix",
                           # "Horas_sueño",
                           # "Dias_ejercicio_semanal",
                           # "Asistencia",
                           # "Salud_mental")]

datos_clasif <- datos[, c("Horas_estudio_diario",
                          "Horas_redes_sociales",
                          "Horas_netflix",
                          "Horas_sueño")]

# Estandarización de las variables
datos_clasif <- scale(datos_clasif)


# Matriz de distancias

prox <- dist(datos_clasif, method = "euclidean")


### CLUSTERING JERÁRQUICO (WARD)

# Aplicación del método de agrupación de Ward
agrupacion <- hclust(prox, method = "ward.D2")

# Representación del dendrograma con colores por grupo
fviz_dend(agrupacion, main = "Dendrograma", k = 4,
          cex = 0.7,
          color_labels_by_k = T,
          rect = T)


# GRUPOS
# Corte del dendrograma en 4 grupos
grupos <- cutree(agrupacion, k = 4)

# Tamaño de cada grupo
table(grupos)

#Representación de los grupos en las dos primeras componentes principales
fviz_cluster(list(data = datos_clasif, cluster = grupos),
             main = "Representación de los grupos",
             ellipse.type = "convex",
             repel = T,
             show.clust.cent = T)


### EVALUACIÓN DE LA AGRUPACIÓN

# Coeficiente de silueta promedio
sil <- silhouette(grupos, dist(datos_clasif))
mean(sil[, 3])

#==================================================
# k-MEDIAS
#==================================================

# Elección del número óptimo de grupos
fviz_nbclust(datos_clasif, kmeans, method = "wss") 

# Tomamos k = 4
kmedias <- kmeans(datos_clasif, centers = 4, nstart = 25)

# Estructura de los grupos
table(kmedias$cluster)

# Visualización en espacio reducido (ACP)
fviz_cluster(kmedias, data = datos_clasif,
             main = "Clustering k-means",
             ellipse.type = "convex",
             palette = "Dark2",
             repel = T,
             show.clust.cent = T)

#Evaluación de la agrupación: Coeficiente de silueta promedio
sil <- silhouette(kmedias$cluster, dist(datos_clasif))
mean(sil[, 3])


# Interpretación de cada grupo

datos_clasif_df <- as.data.frame(datos_clasif)
datos_clasif_df$cluster <- as.factor(kmedias$cluster)
aggregate(. ~ cluster, data = datos_clasif_df, FUN = mean)



#==================================================
# CLASIFICACIÓN SUPERVISADA
#==================================================


### PREPARACIÓN DE LOS DATOS

# Transformamos la variable Calificación a factor con 3 niveles
datos$Rendimiento <- cut(datos$Calificacion,
                         breaks = c(0, 5, 8, 10),
                         labels = c("Bajo", "Medio", "Alto"),
                         right = T)

# Comprobamos la distribución de clases
table(datos$Rendimiento)

# Eliminamos la variable original Calificación 
datos_clasif_sup <- datos [,-15]


# DIVISIÓN ENTRENAMIENTO/PRUEBA

train <- sample(nrow(datos_clasif_sup), 0.7 * nrow(datos_clasif_sup))
test  <- datos_clasif_sup[-train, ]

#==================================================
# ÁRBOL DE DECISIÓN
#==================================================
# Árbol sin restricción (cp = 0) para visualizar el error y seleccionar el cp óptimo
arbol_completo <- rpart(Rendimiento ~ ., data = datos_clasif_sup[train, ],
                        method = "class", cp = 0)

# Gráfico de complejidad para encontrar el cp óptimo
plotcp(arbol_completo)

# Tras observar el gráfico, elegimos un cp que minimice el error de validación cruzada
arbol_podado <- rpart(Rendimiento ~ ., data = datos_clasif_sup[train, ],
                      method = "class", cp = 0.013)

#Visualización del árbol
rpart.plot(arbol_podado,
           extra = 104)


### PREDICCIÓN Y EVALUACIÓN

# Predicción sobre el conjunto de prueba
pred_arbol <- predict(arbol_podado, newdata = test, type = "class")

# Matriz de confusión
confusionMatrix(pred_arbol, test$Rendimiento)


#==================================================
# RANDOM FOREST 
#==================================================

# Entrenamiento del modelo
set.seed(1009)
modelo_rf <- randomForest(Rendimiento ~ ., data = datos_clasif_sup[train, ],ntree = 500, do.trace = 20)

colores <- c("black", "red", "green", "blue")
matplot(modelo_rf$err.rate,  
        type="l",
        xlab="Número de árboles",
        ylab="Error OOB",
        main="Evolución del error")
legend("topright",
       colnames(modelo_rf$err.rate),
       col = colores,
       lty = 1)

#Tomo 140 árboles
set.seed(1009)
modelo_rf <- randomForest(Rendimiento ~ ., data = datos_clasif_sup[train, ],ntree = 140)

### PREDICCIÓN Y EVALUACIÓN

# Predicción sobre el conjunto de prueba
pred_rf <- predict(modelo_rf, newdata = test)

# Matriz de confusión
confusionMatrix(pred_rf, test$Rendimiento)

#Importancia de las variables
varImpPlot(modelo_rf, main = "Importancia de variables (Random Forest)")


#==================================================
# BOOSTING
#==================================================

#Búsqueda de los mejores hiperparámetros
validacion <- trainControl(method = "cv", number = 10)

hiperparametros <- expand.grid(interaction.depth = c(1, 3),
                               n.trees = c(100, 500),
                               shrinkage = c(0.1, 0.01, 0.001),
                               n.minobsinnode = c(1, 10, 20))

set.seed(1009)
modelo_boost <- train(
  Rendimiento ~ ., 
  data = datos_clasif_sup[train, ],
  method = "gbm",
  trControl = validacion,
  tuneGrid = hiperparametros,
  verbose = FALSE
)

modelo_boost$bestTune

# Predicciones
pred_boost <- predict(modelo_boost, newdata = test)

# Matriz de confusión
confusionMatrix(pred_boost, test$Rendimiento)

#Importancia de las variables

imp <- summary(modelo_boost, plotit = F)
ggplot(imp[1:10,],
       aes(x = reorder(var, rel.inf),
           y = rel.inf,
           fill = rel.inf)) +
  geom_col() +
  coord_flip() +
  labs(x = "Variable",
       y = "Influencia relativa",
       title = "Reducción de MSE",
       fill = NULL) +
  theme(legend.position = "bottom")
