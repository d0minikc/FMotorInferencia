using System.Net;
using System;
using Keras;
using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Utils;
using Numpy;
using Numpy.Models;
using System.ComponentModel;
using System.Data;
using System.IO;
using System.Linq;
using System.Security.Cryptography.X509Certificates;

namespace WebApplication3
{
    public class MLModel
    {
        public double Accuracy { get; set; }
        public void TrainAndEvaluate()
        {
            //Descargar el archivo y agregarlo a una dirección cómoda de uso. Adicionalmente, actualizar la ruta al archivo
            string relativePath = Path.Combine("Data", "pima-indians-diabetes.data.csv");

            NDarray dataset = np.loadtxt(relativePath, delimiter: ",");
            NDarray x = np.delete(dataset, 8, axis: 1);
            NDarray y = np.delete(dataset, new int[] { 0, 1, 2, 3, 4, 5, 6, 7 }, axis: 1).reshape(1, dataset.len)[0];

            Sequential model = new Sequential();

            model.Add(new Dense(12, input_shape: (8), activation: "relu"));
            model.Add(new Dense(8, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));
            model.Summary();

            model.Compile(loss: "binary_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            model.Fit(x, y, epochs: 150, batch_size: 150);

            double[] accuracy = model.Evaluate(x, y);
            Accuracy = accuracy[1] * 100;
            Console.WriteLine("Exactitud: " + Accuracy.ToString("0.##") + "%");


        }

    }
}
