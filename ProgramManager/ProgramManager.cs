using Mushroom.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
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
            Network nn = new Network();
            while (iterator < 8124)
            {
                iterator++;
                
                nn.Calculate(reader.ReadNextRow(dataset));  // napisac czytanie jednego wiersza
                int correct = reader.ReadNextClassification(dataset); // czytanie ostaniej kolumny wiersza
                double certainty = -99.0;
                int guess = -1;

                for (int i = 0; i < 10; i++)
                {
                    if (nn.OutputValues[i] > certainty)
                    {
                        certainty = nn.OutputValues[i];
                        guess = i;
                    }
                }
                avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0.0);
                Console.WriteLine("Correct: " + correct + " Correct? " + (guess == correct ? "Yes " : "No ") + " %Correct: " + Math.Round(avg, 10).ToString().PadRight(12)  /* + " Certainty " + Math.Round(certainty, 10)*/ );
                ////nn.Dispose();
            }
        }
        public static void Training()
        {
            reader = new Reader();
            dataset = reader.ReadData();
            for (int i = 0; i < SaveState; i++)
            {
                List<Network> ntw = new List<Network>();
                var tasks = new Task[batchsize];
                for (int j = 0; j < batchsize; j++)
                {
                    //B/c ii may change and this code can't let that happen
                    int copyOfJ = j;
                    double[] row = reader.ReadNextRow(dataset);
                    int correct = reader.ReadNextClassification(dataset);
                    ntw.Add(new Network());
                    tasks[copyOfJ] = Task.Run(() => ntw[copyOfJ].Backprop(row, correct));
                }
                Task.WaitAll(tasks);
                //Syncronously descend
                foreach (Network nn in ntw)
                {
                    nn.Descend();
                    //nn.Dispose();
                }
                //Updating the weights with the avg gradients
                Network.Descend(batchsize);
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

            //Some safety code which is currently disabled
            //if (avgerror > maxavg && iterator > 300) { maxavg = avgerror; }
            //if (avgerror > maxavg * 10 && iterator > 300) { finished = true; }

            //Print various things to the console for verification that things are nominal
            Console.WriteLine("Correct: " + correct + "\t" + " Guess: " + guess + "\t" + " Correct? " + (guess == correct ? "Yes " : "No ").ToString().PadRight(3) + "\t" /*+ "Certainty: " + Math.Round(certainty, 5).ToString().PadRight(7)*/
                + " %Correct: " + "\t" + Math.Round(avg, 5).ToString().PadRight(7) /*+ " Avg error: " + Math.Round(avgerror, 5).ToString().PadRight(8) + " Avg gradient: " + Network.AvgGradient, 15*/);

            //Dispose of the neural network (may not be necessary)
            // nn.Dispose();
            //Reset the console data every few iterations to ensure up to date numbers
            if (iterator > 1000)
            {
                iterator = 100;
                Network.Epoch++;
            }
        }
    }
}
