using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mushroom
{
    public class Reader
    {
        static int resolution = 22;
        static int iterator = 0;

        public double[][] ReadData()
        {
            string[] lines = File.ReadAllLines(@"D:\Semestr 4\Systemy sztucznej inteligencji\REPOZYTORIUM\Mushroom\Mushroom Classification\data\mushroom2.csv");

            double[][] data = new double[lines.Length - 1][];
            for (int i = 1; i < lines.Length; i++)
            {
                string[] tmp = lines[i].Split(',');

                data[i - 1] = new double[tmp.Length];

                for (int j = 1; j < tmp.Length; j++)
                {
                    if (tmp[j].Length == 0)
                    {
                        tmp[j] = "0.1234";
                    }

                    data[i - 1][j] = Convert.ToDouble(tmp[j].Replace('.', ','));
                }
            }
            return data;
        }

        public double[] ReadNextRow(double[][] baseData)
        {
            double[] row = new double[21];


            for (int j = 1; j < baseData[iterator].Length - 2; j++)
            {
                row[j] = baseData[iterator][j];
            }
            
            return row;
        }

        public int ReadNextClassification(double[][] baseData)
        {
            int i = iterator;
            if (iterator < baseData.Length - 1)
            {
                iterator++;
            }
            else
            {
                iterator = 0;
            }
            return Convert.ToInt32(baseData[i].Last());
        }
    }
}
