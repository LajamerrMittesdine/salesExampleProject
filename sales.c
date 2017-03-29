#include <stdio.h>
#include <stdlib.h>
#include "sales.h"

int main(void)
{
    menu_update();
    return 0;
}

void menu(struct customer *cp){
    printf(
    "1. Customer First Name: %s\n"
    "2. Customer Last Name: %s\n"
    "3. Customer Amount: $%.2f\n"
    "4. Customer Card Number: %s\n"
    "5. Customer Card Expiration Month: %i\n"
    "6. Customer Card Expiration Year: %i\n"
    "7. Sales Person: %s\n"
    "If you would like to update a value type the value you would like to update(1-7)\n"
    "If you would like to send this as a email type 8\n"
    "If you would like to clear all customer data type 9\n"
    "If you would like to exit this program type 10\n"
    "Please input an action ID: ",
    cp->first_Name,
    cp->last_Name,
    cp->amount,
    cp->card_Number,
    cp->card_Expiration_Month,
    cp->card_Expiration_Year,
    cp->sales_Person);
}

void cls(void){
    printf("\033c");
}

int get_ID_Input(int a){
    scanf("%i", &a);
    cls();
    return a;
}

void menu_update(void){
    struct customer c = {};
	float f = 0.00;
	int i = 0;
	while (1) {
	cls();
	menu(&c);
	switch(get_ID_Input(0)){
	    case 1:
		printf("Customer First Name: ");
		scanf("%s",c.first_Name);
	        break;
	    case 2:
		printf("Customer Last Name: ");
		scanf("%s",c.last_Name);
	        break;
	    case 3:
		printf("Customer Amount: ");
		scanf("%f",&f);
		c.amount = f;
	        break;
	    case 4:
		printf("Customer Card Number: ");
		scanf("%s",c.card_Number);
	        break;        
	    case 5:
		printf("Customer Card Expiration Month: ");
		scanf("%i",&i);
		c.card_Expiration_Month = i;
	        break;
	    case 6:
		printf("Customer Card Expiration Year: ");
		scanf("%i",&i);
		c.card_Expiration_Year = i;
	        break;
	    case 7:
		printf("Sales Person: ");
		scanf("%s",c.sales_Person);
	        break;
            case 8:
                
                break;
            case 9:
                struct customer c = {};
                break;
	    case 10:
	        exit(0);
	    default:
	        printf("Incorrect action ID!\n");
	        break;
	    }    
	}
}
