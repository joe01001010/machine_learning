/*
Name: Joseph Weibel
Class: 2024Sp CS 2060 002
Due Date: 25APR2024
Professor: Professor Deb Harding
Description of Program: This program will have multiple structures that represent a rideshare. This program will
prompt the admin for login info and allow a set number of attempts. The admin will then input prices for the
ride share. Once the ride share is setup the rider will select which rideshare they want to ride with. The ride
share will then print all ratings from previous riders. After they rider will input how many miles they want to
travel. the input will be validated then the calculations will be made to show prices. Once the prices print
you will see the ride summary. The user will be prompted if they want to leave a rating. The user will enter 'y'
or 'n' and the input will be validated and if the user input y they will give ratings for categories of 
the ride. If the user enters no the program will go back to the beginning for the next rider. If the admin
wants to log in they can enter the sentinel value when prompted for and rider input and they can go back to 
see a summary for all ride shares.
OPERATING SYSTEM=WINDOWS
*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MIN_RAND_MINUTES_FACTOR 1.2
#define MAX_RAND_MINUTES_FACTOR 1.5
#define SENTINEL_NEG1 -1
#define MIN_SET_UP 0.1
#define MAX_SET_UP 50
#define LOGIN_MAX_ATTEMPTS 2
#define MIN_MILES 1
#define MAX_MILES 100
#define SURVEY_RIDER_ROWS 5
#define SURVEY_CATEGORIES 4
#define MIN_RATING 1
#define MAX_RATING 5
#define STRING_LENGTH 80
#define CORRECT_ID "a"
#define CORRECT_PASSCODE "b"
#define FILE_PATH "C:\\rideshare\\"

typedef struct rideShare {
    double baseFare;
    double costPerMinute;
    double costPerMile;
    double minFlatRate;
    char organizationName[STRING_LENGTH];
    double totalMiles;
    int totalMinutes;
    double totalFare;
    int count;
    int ratingsCount;
    int ratings[SURVEY_RIDER_ROWS][SURVEY_CATEGORIES];
    struct rideShare* nextRideShare;
}RideShare;


double getValidDouble(const double min, const double max, const int sentinel);
int getRandomMinutes(double riderMiles, double min_rand_minutes_factor, double max_rand_minutes_factor);
double calculateFare(const RideShare* rideShare, double miles, int minutes);
void printFare(double miles, int minutes, RideShare* rideShare, double fare);
void printSummary(RideShare** rideShareHead, const char* categories[], size_t categoriesSize);
bool adminLogin(const char* passcode, const char* username, const int maxAttempts);
void setupRideShare(RideShare *rideShare);
void displayRideShareDetails(RideShare* rideShare);
void printRatings(const RideShare* rideShare, const char* categories[]);
bool getChoice();
void getUserRating(const char* categories[], RideShare* rideShare, size_t categoriesSize);
int getValidInt(int min, int max);
void removeNewline(char* array);
bool riderMode(RideShare *rideShare, const char* categories[]);
void setupStruct(RideShare *rideShare);
void insertRideShare(RideShare **rideShareHead);
void getUserRideShare(RideShare **rideShareHead, RideShare **rideShareSelected, const char* categories[]);
void freeRemainingRideShares(RideShare** rideShareHead);
void writeSummaryToFile(RideShare** rideShareHead, const char* categories[]);

int main(void) {
    RideShare *rideShareHeadPtr = NULL;
    RideShare* rideShareSelected = NULL;
    const char* categories[] = { "1. Safety", "2. Cleanliness", "3. Comfort", "4. Test" };
    bool riderModeVariable = true;

    //Sending constant username and password to compare with what the user enters
    //Sending MAX_ATTEMPTS to use in the while loop to kill the program after pre defined
    // number of failed attempts
    /*
    * Returns: boolean
    * Parameters: const address of index 0 char array, const address of index 0 char array, const int
    * Description: This function will prompt for id and psscode and match them to the paramaters passes as 
    * correct ID and correct passcode. The function will iterate to the max attempts then return false if
    * no valid input is entered else it will return true
    */
    bool adminLoginResult = adminLogin(CORRECT_PASSCODE, CORRECT_ID, LOGIN_MAX_ATTEMPTS);
    if (adminLoginResult) {
        /*
        * Returns: void
        * Parameters: double pointer
        * Description: This function will take the rideShare pointer head and ask if the admin wants to enter
        * another rideshare. This will then call other function to setup the rideshares. This function will
        * create a linked list and insert the new rideshares in alphabetical order.
        */
        insertRideShare(&rideShareHeadPtr);

        while (riderModeVariable) {
            /*
            * Returns: void
            * Parameters: double pointer RideShare, double pointer RideShare, const char[]
            * Description: This function will display the rideshares and let the user select which rideshare they want to use.
            * This function will update rideShareSelected with the address of the selected rideshare
            */
            getUserRideShare(&rideShareHeadPtr, &rideShareSelected, categories);
            /*
            * Returns: bool
            * Parameters: double pointer RideShare, const char[]
            * Description: This function is the main logic flow. This will ask the user how many mile,
            * make the calculations and get a user rating from the user, this will then return a bool to
            * let the program know to prompt for a new rider
            */
            riderModeVariable = riderMode(rideShareSelected, categories);
        }
        /*
        * Returns: void
        * Parameters: double pointer RideShare, const char[], const int
        * Description: This function will iterate over the rideShares in the linked list and print a summary of each to the console
        */
        printSummary(&rideShareHeadPtr, categories, SURVEY_CATEGORIES);
        /*
        * Returns: void
        * Parameters: double pointer RideShare, const char[]
        * Description: This function will iterate over the rideShares in the linked list and write a summary of each to a file
        */
        writeSummaryToFile(&rideShareHeadPtr, categories);
    }//end if(adminLoginResult)
    /*
    * Returns: void
    * Parameters: double pointer RideShare
    * Description: This function will iterate over the rideShares in the linked list and free all the memory on the heap
    */
    freeRemainingRideShares(&rideShareHeadPtr);
}//End main

bool adminLogin(const char *passcode, const char *username, int maxAttempts) {
    bool validLogin = false;
    char user[80] = { 0 };
    char pass[80] = { 0 };
    int attempt = 0;
    puts("Admin Login\n");

    while (!validLogin && attempt < maxAttempts) {
        puts("Enter your Admin id.");
        //Because using fgets need to save the stdin as a string to convert to int
        //This function expects a character array, length of the array, and how it will receive input
        fgets(user, sizeof(user), stdin);

        puts("Enter your Admin passcode.");
        //Because using fgets need to save the stdin as a string to convert to int
        //This function expects a character array, length of the array, and how it will receive input
        fgets(pass, sizeof(pass), stdin);

        /*
        * Returns: void
        * Parameters: char[]
        * Description: This function expects an array of characters and will check if strlen-1 is a '\n' and change it to '\0'
        */
        removeNewline(user);
        removeNewline(pass);

        /*
        * Returns: bool
        * Parameters: char[], char[]
        * Description: This function expects an array of characters and will check if strlen-1 is a '\n' and change it to '\0'
        * call STRCMP and pass the user generated values to the comparison values.
        * This function will return a boolean value. The ! is so the value returns the proper boolean value.
        * strcmp will return 1 which is false if they are matches.
        */
        bool goodUser = !strcmp(user, username);
        bool goodPass = !strcmp(pass, passcode);

        if (goodUser && goodPass) {
            validLogin = true;
        }
        if (!validLogin) {
            attempt++;
            if (attempt >= maxAttempts) {
                puts("Exiting RideShare");
            }
            else {
                puts("Invalid username or password. Please try again.");
            }
        }
    }
    return validLogin;
}

void setupRideShare(RideShare *rideShare) {
    puts("Setup rideshare information");
    
    int badSentinel = -12345;
    puts("Enter the base fare");
    /*
    * Returns: double
    * Parameters: const double, const double, const int
    * Description: Get valid double expects a min, max, and sentinel value. This will then return a double
    */
    rideShare->baseFare = getValidDouble(MIN_SET_UP, MAX_SET_UP, badSentinel);

    puts("Enter the cost per minute");
    rideShare->costPerMinute = getValidDouble(MIN_SET_UP, MAX_SET_UP, badSentinel);

    puts("Enter the cost per mile");
    rideShare->costPerMile = getValidDouble(MIN_SET_UP, MAX_SET_UP, badSentinel);
    
    puts("Enter the minimum flat rate");
    rideShare->minFlatRate = getValidDouble(MIN_SET_UP, MAX_SET_UP, badSentinel);
    
    puts("Enter the organization name");
    /*
    * Returns: pointer
    * Parameters: char[], size_t length of char array, method of input
    * Description: expects a second argument of the sizeof what the character array is, thats why sizeof() works here
    */
    fgets(rideShare->organizationName, sizeof(rideShare->organizationName), stdin);
    /*
    * Returns: void
    * Parameters: char[]
    * Description: This function expects an array of characters and will check if strlen-1 is a '\n' and change it to '\0'
    */
    removeNewline(rideShare->organizationName);
    /*
    * Returns: void
    * Parameters: pointer to RideShare
    * Description: this function expects a pointer to a struct as an argument and will print the values that define the
    * Rideshare that the admin just set. This is a void function and doesn't return anything
    */
    displayRideShareDetails(rideShare);
}

void printRatings(const RideShare *rideShare, const char* categories[]) {
    size_t categoryCount = SURVEY_CATEGORIES;
    size_t rowCount = rideShare->ratingsCount;
    printf("RideShare Organization %s ratings\n", rideShare->organizationName);
    puts("Survey Results");

    if (rideShare->count == 0) {
        puts("No Ratings Currently");
    }
    else {
        printf("Rating Categories:\t");
        for (size_t category = 0; category < categoryCount; category++) {
            printf("%s\t", categories[category]);
        }
        puts("");

        for (size_t row = 0; row < rowCount; row++) {
            printf("Survey: %zu\t\t", row+1);
            for (size_t column = 0; column < categoryCount; column++) {
                printf("%d\t\t", rideShare->ratings[row][column]);
            }
            puts("");
        }
    }
    puts("");
}

double getValidDouble(const double min, const double max, const int sentinel) {
    bool goodInput = false;
    double numberToReturn = 0;
    char stringToConvert[STRING_LENGTH] = { '\0' };
    char* endOfDoublePtr = NULL;

    //Using a while loop with good input initialized to false because I think it reads easier
    while (!goodInput) {
        //Because using fgets need to save the stdin as a string to convert to int
        //This function expects a character array, length of the array, and how it will receive input
        fgets(stringToConvert, STRING_LENGTH, stdin);
        /*
        * Returns: void
        * Parameters: char[]
        * Description: This function expects an array of characters and will check if strlen-1 is a '\n' and change it to '\0'
        */
        removeNewline(stringToConvert);
        /*
        * Returns: double
        * Parameters: char[], pointer to double
        * Description: strtod expects the string to be converted as a charactrer array and how to determine the end of the string. This will return a double
        */
        numberToReturn = strtod(stringToConvert, &endOfDoublePtr);

        //Check if sentinel value was entered and exit loop to exit function
        if (numberToReturn == sentinel) {
            goodInput = true;
        }
        else {
            if (*endOfDoublePtr != '\0') {
                printf("Error: Enter a number from %.1lf to %.1lf\n", min, max);
            }
            else {
                //Check if the decimal entered is between the min and max
                if (numberToReturn >= min && numberToReturn <= max) {
                    goodInput = true;
                }
                else {
                    printf("Error: Enter a number from %.1lf to %.0lf\n", min, max);
                }
            }
        }
    }
    return numberToReturn;
}

int getRandomMinutes(double riderMiles, double min_rand_minutes_factor, double max_rand_minutes_factor) {
    //Give srand a time seed so it can generate a new value everytime the code is ran
    srand(time(NULL));
    double minRandomMinutes = riderMiles * min_rand_minutes_factor;
    double maxRandomMinutes = riderMiles * max_rand_minutes_factor;
    //Formula to calculate a random number between the min and max
    double randomMinutes = minRandomMinutes + ((float)rand() / RAND_MAX) * (maxRandomMinutes - minRandomMinutes);
    //Type casting from double to int
    int randomMinutesInt = (int)randomMinutes;
    return randomMinutesInt;
}

double calculateFare(const RideShare *rideShare, double miles, int minutes) {
    //Formula for calculating fare as described in the assignment requirements
    double fare = rideShare->baseFare + (rideShare->costPerMinute * minutes) + (rideShare->costPerMile * miles);

    //Make sure the fare is above the minRate
    if (fare < rideShare->minFlatRate) {
        fare = rideShare->minFlatRate;
    }

    return fare;
}

void printFare(double miles, int minutes, RideShare *rideShare, double fare) {
    puts("Current Ride Information");
    puts("Thanks for riding with us\n");
    puts("Rider\t\tNumber of Miles\t\tNumber of Minutes\t\tRide Fare Amount");
    printf("%d\t\t\t%.1f\t\t\t%d\t\t\t%.2f\n\n", rideShare->count, miles, minutes, fare);
}

void printSummary(RideShare** rideShareHead, const char* categories[], size_t categoriesSize) {
    double averagesArray[SURVEY_CATEGORIES] = { 0 };
    size_t averagesArraySize = SURVEY_CATEGORIES;
    size_t indexRatings = SURVEY_RIDER_ROWS;

    RideShare* rideShare = *rideShareHead;

    while (rideShare != NULL) {
        if (rideShare->count == 0) {
            printf("\n%s Summary Report\n", rideShare->organizationName);
            puts("There were no rides");
        }
        else {
            for (size_t category = 0; category < averagesArraySize; category++) {
                for (size_t rating = 0; rating < indexRatings; rating++) {
                    averagesArray[category] += rideShare->ratings[rating][category];
                }
                averagesArray[category] = averagesArray[category] / rideShare->ratingsCount;
            }


            printf("\n%s Summary Report\n", rideShare->organizationName);
            puts("Rider  	Number of Miles     Number of Minutes    Ride Fare Amount");
            printf("%d\t\t%.1f\t\t%d\t\t%.2f\n", rideShare->count, rideShare->totalMiles, rideShare->totalMinutes, rideShare->totalFare);

            puts("Category Rating Averages");
            for (size_t category = 0; category < averagesArraySize; category++) {
                printf("%s\t", categories[category]);
            }
            puts("");
            for (size_t category = 0; category < averagesArraySize; category++) {
                printf("%.2f\t\t", averagesArray[category]);
            }
        }
        rideShare = rideShare->nextRideShare;
        puts("");
    }
    puts("\n\nExiting RideShare Program\n");
}

void displayRideShareDetails(RideShare *rideShare) {
    puts("");
    printf("%s\n", rideShare->organizationName);
    printf("baseFare = $%.2lf\n", rideShare->baseFare);
    printf("costPerMinute = $%.2lf\n", rideShare->costPerMinute);
    printf("costPerMile = $%.2lf\n", rideShare->costPerMile);
    printf("minFlatRate = $%.2lf\n", rideShare->minFlatRate);
}

bool getChoice() {
    bool ride = false;
    bool validChoice = false;
    char tempChoice[STRING_LENGTH] = { 0 };

    while (!validChoice) {
        /*
        * Returns: pointer to char[]
        * Parameters: char[], int, input method
        * Description: Because using fgets need to save the stdin as a string to convert to int. This function expects a character array, length of the array, and how it will receive input
        */
        fgets(tempChoice, STRING_LENGTH, stdin);
        char choice = tempChoice[0];


        if (choice == 'y' || choice == 'Y') {
            validChoice = true;
            ride = true;
        }
        else if (choice == 'n' || choice == 'N') {
            validChoice = true;
        }
        else {
            puts("You did not enter a y or n.");
        }
    }
    return ride;
}

void getUserRating(const char *categories[], RideShare *rideShare, size_t categoriesSize) {
    puts("We want to know how your experience was on your ride today.Using the rating system 1 to 5 enter your rating for each category:");
    for (size_t category = 0; category < categoriesSize; category++) {
        puts("Enter your rating for");
        printf("%s\n", categories[category]);
        /*
        * Returns: int
        * Parameters: int, int
        * Description: Get valid int expects a min and max int and will return an int
        */
        rideShare->ratings[rideShare->ratingsCount - 1][category] = getValidInt(MIN_RATING, MAX_RATING);
    }
}

int getValidInt(int min, int max) {
    bool goodRating = false;
    int rating = 0;
    char* endOfLongPtr = NULL;
    char stringToConvert[STRING_LENGTH] = { 0 };

    while (!goodRating) {
        /*
        * Returns: pointer to char array
        * Parameters: address of char[], int for string length, input method
        * Description: Because using fgets need to save the stdin as a string to convert to int
        * This function expects a character array, length of the array, and how it will receive input
        */
        fgets(stringToConvert, STRING_LENGTH, stdin);
        /*
        * Returns: void
        * Parameters: char[]
        * Description: This function expects an array of characters and will check if strlen-1 is a '\n' and change it to '\0'
        */
        removeNewline(stringToConvert);
        /*
        * Returns: int
        * Parameters: char[], pointer, base of number
        * Description: This cuntion is taking the character array as an argument, how to detect the end of the input, and
        * The base we are working in. This will then be casted to an int and stored in a variable
        */
        rating = (int)strtol(stringToConvert, &endOfLongPtr, 10);

        if (rating >= 1 && rating <= 5) {
            goodRating = true;
        }
        else {
            puts("Enter a number 1-5");
        }
    }
    return rating;
}

void removeNewline(char *array) {
    //Checking if the last index of the string length is newline, and changing it to null terminator
    if (array[strlen(array) - 1] == '\n') {
        array[strlen(array) - 1] = '\0';
    }
}

bool riderMode(RideShare *rideShare, const char* categories[]) {
    bool giveRating = false;
    bool riderMode = true;
    double riderMiles = 0;
    int riderMinutes = 0;
    double riderFare = 0;

    if (rideShare->count < SURVEY_RIDER_ROWS) {
        /*
        * Returns: void
        * Parameters: pointer RideShare, char[][]
        * Description: Passing pointers to the riderShare struct and the totals struct passing a pointer to an array of categories and ratings which is a 2d array
        */
        printRatings(rideShare, categories);


        puts("Enter the number of miles to your destination.");
        /*
        * Returns: double
        * Parameters: const double, const double, const int
        * Description: Get valid double expects a min, max, and sentinel value. This will then return a double
        */
        riderMiles = getValidDouble(MIN_MILES, MAX_MILES, SENTINEL_NEG1);

        if (riderMiles == SENTINEL_NEG1) {
            bool adminLoginResult = adminLogin(CORRECT_PASSCODE, CORRECT_ID, LOGIN_MAX_ATTEMPTS);
            if (adminLoginResult) {
                riderMode = false;
            }
        }
        else {
            /*
            * Returns: int
            * Parameters: double, double, double
            * Description: This function takes the riderMiles and the constant min random factor and max 
            * and will return an integer representing the randomly generated minutes the ride will take
            */
            riderMinutes = getRandomMinutes(riderMiles, MIN_RAND_MINUTES_FACTOR, MAX_RAND_MINUTES_FACTOR);
            /*
            * Returns: double
            * Parameters: pointer to rideshare, double, double
            * Description: This function will take in the rideShare struct address, riderMiles as a double, 
            * and riderMinutes as an int. This function will return a double of the fare for the ride.
            */
            riderFare = calculateFare(rideShare, riderMiles, riderMinutes);

            //Incremenet the totals
            rideShare->count++;
            rideShare->totalFare += riderFare;
            rideShare->totalMiles += riderMiles;
            rideShare->totalMinutes += riderMinutes;

            /*
            * Returns: void
            * Parameters: double, int, pointer to rideshare, double
            * Description: This function will print the cost and length in time of the ride and distance
            */
            printFare(riderMiles, riderMinutes, rideShare, riderFare);

            puts("Do you want to rate your rideshare experience?");
            /*
            * Returns: boolean
            * parameters: None
            * Description: will prompt user for y or n and return boolean based on choice
            */
            giveRating = getChoice();

            if (giveRating) {
                //This function is void
                //This function expects an array of pointers to char, 2d array to ints, struct point
                //and constant that will be size_t used for iterating
                rideShare->ratingsCount++;
                /*
                * Returns: void
                * Parameters: char[][], pointer to rideshare, char[][]
                * Description: This prompts the user to enter 1-5 for a rating of the rideshare in each category
                */
                getUserRating(categories, rideShare, SURVEY_CATEGORIES);

            }
            else {
                puts("Thanks for riding with us.");
            }
        }//End else if rider input is not sentinel value
    }//End (rideShare->count < SURVEY_RIDER_ROWS)
    else {
        puts("Not enough room for another rider.");
    }
    return riderMode;
}//End Rider mode function

void setupStruct(RideShare* rideShare) {
    rideShare->baseFare = 0;
    rideShare->costPerMile = 0;
    rideShare->costPerMinute = 0;
    rideShare->count = 0;
    rideShare->minFlatRate = 0;
    for (size_t index = 0; index < sizeof(rideShare->organizationName); index++) {
        rideShare->organizationName[index] = '\0';
    }
    rideShare->ratingsCount = 0;
    rideShare->totalFare = 0;
    rideShare->totalMiles = 0;
    rideShare->totalMinutes = 0;
    for (size_t rating = 0; rating < SURVEY_RIDER_ROWS; rating++) {
        for (size_t category = 0; category < SURVEY_CATEGORIES; category++) {
            rideShare->ratings[rating][category] = 0;
        }
    }
    rideShare->nextRideShare = NULL;
}

void insertRideShare(RideShare** rideShareHead) {
    bool choice = true;

    while (choice) {
        puts("Do you want to add another rideshare, y or n?");
        /*
        * Returns: boolean
        * parameters: None
        * Description: will prompt user for y or n and return boolean based on choice
        */
        choice = getChoice();

        if (choice) {
            /*
            * Returns: pointer to memory allocated
            * Parameters: size_t of the struct data that needs to be stored in memory
            * Description: malloc will allocate memory for a struct on the heap
            */
            RideShare* newRideShare = malloc(sizeof(RideShare));
            if (newRideShare != NULL) {
                /*
                * Returns: void
                * Parameters: pointer to struct
                * Description: Will initialize all struct values to 0 or NULL
                */
                setupStruct(newRideShare);
                /*
                * Returns: void
                * Parameters: pointer to struct
                * Description: Will initialize all struct values to what the user inputs
                */
                setupRideShare(newRideShare);
                RideShare* previousRideShare = NULL;
                RideShare* currentRideShare = *rideShareHead;

                while (currentRideShare != NULL && strcmp(newRideShare->organizationName, currentRideShare->organizationName) > 0){
                    previousRideShare = currentRideShare;
                    currentRideShare = currentRideShare->nextRideShare;
                }

                if (previousRideShare == NULL) {
                    *rideShareHead = newRideShare;
                }
                else {
                    previousRideShare->nextRideShare = newRideShare;
                }
                newRideShare->nextRideShare = currentRideShare;

            }
            else {
                puts("Memory allocation failed");
            }
        }
    }
    puts("\nExiting Admin Mode\n");
}

void getUserRideShare(RideShare **rideShareHead, RideShare **rideShareSelected, const char* categories[]) {
    char orgToReturn[STRING_LENGTH] = { '\0' };
    bool goodInput = false;

    RideShare* currentRideShare = *rideShareHead;

    while (currentRideShare != NULL) {
        /*
        * Returns: void
        * Parameters: pointer RideShare, char[][]
        * Description: Passing pointers to the riderShare struct and the totals struct passing a pointer to an array of categories and ratings which is a 2d array
        */
        printRatings(currentRideShare, categories);
        currentRideShare = currentRideShare->nextRideShare;
    }

    while (!goodInput) {
        RideShare* currentRideShare = *rideShareHead;
        puts("Enter the name of the ride share you want to use.");
        fgets(orgToReturn, STRING_LENGTH, stdin);
        /*
        * Returns: void
        * Parameters: char[]
        * Description: This function expects an array of characters and will check if strlen-1 is a '\n' and change it to '\0'
        */
        removeNewline(orgToReturn);

        while (currentRideShare != NULL && !goodInput) {
            if (strcmp(orgToReturn, currentRideShare->organizationName) != 0) {
                currentRideShare = currentRideShare->nextRideShare;
            }
            else {
                goodInput = true;
                *rideShareSelected = currentRideShare;
            }
        }
        if (!goodInput) {
            puts("Error, the ride share you entered doesn't match.");
        }
    }
}

void freeRemainingRideShares(RideShare** rideShareHead)
{
    RideShare* currentRideShare = *rideShareHead;
    RideShare* nextRideShare = NULL;

    while (currentRideShare != NULL)
    {
        nextRideShare = currentRideShare->nextRideShare;
        /*
        * Returns: NULL
        * Parameters: pointer to rideshare
        * Description: This function will make the pointer to the memory address null on the heap
        */
        free(currentRideShare);
        currentRideShare = nextRideShare;
    }

    *rideShareHead = NULL;
}

void writeSummaryToFile(RideShare** rideShareHead, const char* categories[]) {
    RideShare* currentRideShare = *rideShareHead;
    double averagesArray[SURVEY_CATEGORIES] = { 0 };
    size_t averagesArraySize = SURVEY_CATEGORIES;
    size_t indexRatings = SURVEY_RIDER_ROWS;

    while (currentRideShare != NULL) {
        FILE* currentRideShareFile = NULL;
        char filePath[STRING_LENGTH] = FILE_PATH;
        /*
        * Returns: void
        * Parameters: char*destingation,. const char *source
        * Description: Thsi function joins two strings into one string and makes them the first variable
        */
        strcat(filePath, currentRideShare->organizationName);
        strcat(filePath, ".txt");

        if ((currentRideShareFile = fopen(filePath, "w")) == NULL) {
            puts("File could not be opened");
        }
        else {
            if (currentRideShare->count == 0) {
                fprintf(currentRideShareFile, "\n%s Summary Report\n", currentRideShare->organizationName);
                fputs("There were no rides", currentRideShareFile);
            }
            else {
                for (size_t category = 0; category < averagesArraySize; category++) {
                    for (size_t rating = 0; rating < indexRatings; rating++) {
                        averagesArray[category] += currentRideShare->ratings[rating][category];
                    }
                    averagesArray[category] = averagesArray[category] / currentRideShare->ratingsCount;
                }


                fprintf(currentRideShareFile, "%s Summary Report\n", currentRideShare->organizationName);
                fputs("Rider  	Number of Miles     Number of Minutes    Ride Fare Amount\n", currentRideShareFile);
                fprintf(currentRideShareFile, "%d\t\t%.1f\t\t%d\t\t%.2f\n", currentRideShare->count, currentRideShare->totalMiles, currentRideShare->totalMinutes, currentRideShare->totalFare);

                fputs("\nCategory Rating Averages\n", currentRideShareFile);
                for (size_t category = 0; category < averagesArraySize; category++) {
                    fprintf(currentRideShareFile, "%s\t", categories[category]);
                }
                fputs("\n", currentRideShareFile);
                for (size_t category = 0; category < averagesArraySize; category++) {
                    fprintf(currentRideShareFile, "%.2f\t\t", averagesArray[category]);
                }
            }
        }
        currentRideShare = currentRideShare->nextRideShare;
    }
}