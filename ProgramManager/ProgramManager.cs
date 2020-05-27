using Mushroom.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace Mushroom
{
    public class ProgramManager
    {
        public static bool isrunning = false;
        public static bool finished = false;
        static double SaveState = 100;
        static double avg = 0;
        //static double maxavg = 0;
        static double avgerror = 0;
        static int batchsize = 5;
        static int iterator = 0;
        static double[][] dataset;
        static Reader reader;
        static DataDivison datadivision= new DataDivison();
        static int lengthOfTrainigData;        //calling the function that counts the size of training data

        public ProgramManager()
        {
            
        }

        //Reset (re-initialize) weights and biases of the neural network
        public static void reset()
        {
            Network nn = new Network();
            nn.initialize();
            NetworkParametersHandler.WriteWeightBias();
        }
        public static void Testing()
        {
            int correctAnswersCount = 0;
            reader = new Reader();
            double[][] dataset = reader.ReadData();
            lengthOfTrainigData = datadivision.GetPartOfData(70, dataset);
            //double[][] testingData = datadivision.Skip(dataset, lengthOfTrainigData);    //creating an array of test data in the right size. The skip function skips the amount of data given as an argument and returns the others
            double[][] testingData =dataset.Skip(lengthOfTrainigData).ToArray();    //creating an array of test data in the right size. The skip function skips the amount of data given as an argument and returns the others
            Network nn = new Network();
            while (iterator < testingData.Length)
            {
                iterator++;
                
                nn.Calculate(reader.ReadNextRow(testingData));  // napisac czytanie jednego wiersza
                int correct = reader.ReadNextClassification(testingData); // czytanie ostaniej kolumny wiersza
                double certainty = -99.0;
                int guess = -1;

                for (int i = 0; i < 2; i++)
                {
                    if (nn.OutputValues[i] > certainty)
                    {
                        certainty = nn.OutputValues[i];
                        guess = i;
                    }
                }
                avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0.0);
                Console.WriteLine("Correct: " + correct + " Correct? " + (guess == correct ? "Yes " : "No "));
                if (guess == correct)
                {
                    correctAnswersCount++;
                }
            }
            Console.WriteLine("Percentage of correct answers = " + ((double)correctAnswersCount/(double)iterator)*100.0);
        }
        public static void Training()
        {
            reader = new Reader();
            dataset = reader.ReadData();
            lengthOfTrainigData = datadivision.GetPartOfData(70, dataset);
            double[][] trainingData = datadivision.Take(dataset, lengthOfTrainigData);         //creating a board with training data in the right size
            for (int i = 0; i < SaveState; i++)
            {
                List<Network> ntw = new List<Network>();
                var tasks = new Task[batchsize];
                for (int j = 0; j < batchsize; j++)
                {
                    //B/c ii may change and this code can't let that happen
                    int copyOfJ = j;
                    double[] row = reader.ReadNextRow(trainingData);
                    int correct = reader.ReadNextClassification(trainingData);
                    ntw.Add(new Network());
                    tasks[copyOfJ] = Task.Run(() => ntw[copyOfJ].Backprop(row, correct));
                }
                Task.WaitAll(tasks);
                //Syncronously descend
                foreach (Network nn in ntw)
                {
                    nn.Descent();
                    //nn.Dispose();
                }
                //Updating the weights with the avg gradients
                Network.Descent(batchsize);
                UserValidation();
            }
            //Save weights and biases
            NetworkParametersHandler.WriteWeightBias();
        }
        public static void UserValidation()
        {
            Network nn = new Network();
            double[] row;
            int correct;

            //Some user validation code

            row = reader.ReadNextRow(dataset);       //ta sama akcja co wyzej
            correct = reader.ReadNextClassification(dataset);
            //Backprop again for averaging
            nn.Calculate(row);


            int guess = 0;
            double certainty = 0;

            for (int i = 0; i < 2; i++)
            {
                if (nn.OutputValues[i] > certainty) //finding the biggest value in output layer
                {
                    certainty = nn.OutputValues[i];
                    guess = i;
                }
            }
            //Calculate the moving average of the percentage of trials correct of those written to console
            double error = 0;
            for (int i = 0; i < Network.OutputCount; i++)
            {
                error += ((i == correct ? 1.0 : 0.0) - nn.OutputValues[i]) * ((i == correct ? 1.0 : 0.0) - nn.OutputValues[i]);
            }
            iterator++;
            avgerror = (((double)iterator / ((double)iterator + 1)) * avgerror) + ((1 / (double)iterator) * error);
            avg = (avg * ((double)iterator / ((double)iterator + 1))) + ((guess == correct) ? (1 / (double)iterator) : 0.0);

            //Print various things to the console for verification that things are nominal
            Console.WriteLine("Correct: " + correct + "\t" + " Guess: " + guess + "\t" + " Correct? " + (guess == correct ? "Yes " : "No ").ToString().PadRight(3) + "\t"  + " %Correct: " + "\t" + Math.Round(avg, 5).ToString().PadRight(7));

            //Reset the console data every few iterations to ensure up to date numbers
            if (iterator > 1000)
            {
                iterator = 100;
                Network.Epoch++;
            }
        }
    }
}
