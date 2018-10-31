#include "mvlar.h"
#include <navdata_client.h>

using namespace std;

int main()
{
	cout << "Hello CMake." << endl;

	navdata_client nav_cli;
	nav_cli.Join();
	
	return 0;
}
