#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include "keras_model.H"

using namespace std;

//function for reading input variables from a file
double readFromFile(string file_name, string keyword)
{
  string key;
  string value;
  ifstream file;

  file.open(file_name);

  bool found_keyword = false;
  while(getline(file,key,',') && !found_keyword)
    {
      getline(file,value,'\n');
      if(keyword == key)
        {
          found_keyword = true;
        }
    }
  file.close();
  return atof(value.c_str());
}



int main ()
{
  //read flow variables from the input file "input.csv"
  double solid_volume_fraction = readFromFile("input.csv", "solid_volume_fraction");
  double filter_size = readFromFile("input.csv", "filter_size");
  double grad_P = readFromFile("input.csv", "grad_P");
  double slip_velocity = readFromFile("input.csv", "slip_velocity");
  double Reynolds_number = readFromFile("input.csv", "Reynolds_number");

  //read characteristic parameters from the input file "input.csv"
  double particle_diameter = readFromFile("input.csv", "particle_diameter");
  double terminal_velocity = readFromFile("input.csv", "terminal_velocity");
  double maximum_solid_volume_fraction = readFromFile("input.csv", "maximum_solid_volume_fraction");
  double particle_density = readFromFile("input.csv", "particle_density");
  double g = 9.81;
  
  //nondimensionalize/scale the flow variables for model input
  double Froude_number = pow(terminal_velocity, 2)/particle_diameter/g;
  double scaled_solid_volume_fraction = solid_volume_fraction/maximum_solid_volume_fraction;
  double dimless_filter_size = particle_diameter*pow(Froude_number, 1.0/3.0)/filter_size;
  double dimless_slip_velocity = slip_velocity/terminal_velocity;
  double dimless_grad_P = grad_P/particle_density/g;
 

  //define number of features to be fed into neural network model
  int num_features = 5;

  //read normalizing parameters for features in "mean.csv" and "std.csv"
  double* means_;
  double* stds_;
  means_ = new double[num_features];
  stds_ = new double[num_features];
  ifstream reader;
  reader.open("Model/mean.csv");
  for (int i=0; i<num_features; i++)
    reader>>means_[i];
  reader.close();

  reader.open("Model/std.csv");
  for (int i=0; i<num_features; i++)
    reader>>stds_[i];
  reader.close();


  //define model objects and compute prediction for drift_flux
  keras::KerasModel DFnnModel_("Model/DFkerasParameters.nnet", false);
  keras::DataChunkFlat dataChunk( num_features, 0.0);
  vector<float>& features = dataChunk.get_1d_rw(); 
  float scaled_dimless_drift_flux = 1.0e-6;
  float drift_flux = 1.0e-6;
  //collect features and normalize the features using mean and std input
  unsigned int index = 0;
  features.clear();
  features.push_back((Reynolds_number - means_[index])/stds_[index]);
  ++index;
  features.push_back((dimless_filter_size - means_[index])/stds_[index]);
  ++index;
  features.push_back((scaled_solid_volume_fraction - means_[index])/stds_[index]);
  ++index;
  features.push_back((dimless_grad_P - means_[index])/stds_[index]);
  ++index;
  features.push_back((dimless_slip_velocity - means_[index])/stds_[index]);
  
  //compute prediction
  vector<float> prediction = DFnnModel_.compute_output( &dataChunk );
  scaled_dimless_drift_flux = prediction.at(0);
  drift_flux = scaled_dimless_drift_flux*maximum_solid_volume_fraction*terminal_velocity;

  //output variables and prediction
  cout << "---input variables ---" << endl;
  cout << "Reynolds number: " << Reynolds_number << endl;
  cout << "solid volume fraction: " << solid_volume_fraction << endl;
  cout << "filter size: " << filter_size << endl;
  cout << "grad_P: " << grad_P << endl;
  cout << "slip_velocity: " << slip_velocity << endl;
  cout << "particle_diameter: " << particle_diameter << endl;
  cout << "scaled solid volume fraction: " << scaled_solid_volume_fraction << endl;
  cout << "dimless filter size: " << dimless_filter_size << endl;
  cout << "dimless grad P: " << dimless_grad_P << endl;
  cout << "dimless_slip_velocity: " << dimless_slip_velocity << endl;
  cout << "Fr: " << Froude_number << endl;
  cout << "---prediction ---" << endl;
  cout << "prediction: " << drift_flux << endl;  
}

